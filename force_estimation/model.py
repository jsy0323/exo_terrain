"""Symmetric IMU token encoding, masked pooling and six causal TCN layers."""

import math
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import EXTRA_NAMES


class CausalLayer(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        # LayerNorm acts within one time instant, never across the time axis.
        residual = self.conv(F.pad(sequence.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return sequence + self.dropout(F.elu(self.norm(residual)))


class MomentEstimator(nn.Module):
    receptive_field = 127

    def __init__(self, sample_rate_hz=100, window_s=1.0, extra_dim=len(EXTRA_NAMES),
                 dropout=0.1, torque_scale_nm=50.0, grf_scale_n=750.0):
        super().__init__()
        self.window_steps = math.ceil(float(sample_rate_hz) * float(window_s))
        if self.window_steps < 1 or self.window_steps > self.receptive_field:
            raise ValueError(f"The six-layer TCN supports 1..127 samples, requested {self.window_steps}")
        self.torque_scale_nm = float(torque_scale_nm)
        self.grf_scale_n = float(grf_scale_n)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(9, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU()) for _ in range(4)
        ])
        self.part_embedding = nn.Parameter(torch.zeros(7, 64))
        nn.init.normal_(self.part_embedding, std=0.02)
        self.attention = nn.Linear(64, 1)
        self.input_projection = nn.Linear(64 + extra_dim, 128)
        self.layers = nn.ModuleList([CausalLayer(128, dilation, dropout) for dilation in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Linear(128, 16)
        # Gyro is an axial vector. Acceleration and gravity are polar vectors.
        self.register_buffer("right_mirror", torch.tensor([-1., 1., -1., 1., -1., 1., 1., -1., 1.]))
        self.register_buffer("imu_scale", torch.tensor([5., 5., 5., 20., 20., 20., 1., 1., 1.]))
        self.register_buffer("extra_mean", torch.zeros(extra_dim))
        self.register_buffer("extra_std", torch.ones(extra_dim))
        self.register_buffer("uncertainty_scale", torch.ones(2), persistent=False)

    def forward_sequence(self, imu, imu_mask, extra):
        if imu.shape[-2:] != (7, 9) or imu_mask.shape != imu.shape[:-1] or extra.shape[:-1] != imu.shape[:-2]:
            raise ValueError("Expected imu[B,T,7,9], mask[B,T,7], extra[B,T,D]")
        mask = imu_mask.bool()
        clean = torch.where(mask.unsqueeze(-1), imu, torch.zeros_like(imu))
        tokens = []
        for part, encoder_index in enumerate((0, 1, 1, 2, 2, 3, 3)):
            value = clean[..., part, :]
            if part in (2, 4, 6):
                value = value * self.right_mirror
            tokens.append(self.encoders[encoder_index](value / self.imu_scale) + self.part_embedding[part])
        tokens = torch.stack(tokens, dim=-2)
        scores = self.attention(tokens).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = (weights.unsqueeze(-1) * tokens).sum(dim=-2)
        hidden = self.input_projection(torch.cat((fused, (extra - self.extra_mean) / self.extra_std), dim=-1))
        for layer in self.layers:
            hidden = layer(hidden)
        output = self.head(hidden)
        sigma_normalized = F.softplus(output[..., 14:16]) + 1e-3
        sigma_nm = sigma_normalized * self.torque_scale_nm * self.uncertainty_scale
        return {
            "tau_hip": output[..., :2] * self.torque_scale_nm,
            "tau_aux": output[..., 2:6] * self.torque_scale_nm,
            "grf_v": output[..., 6:8] * self.grf_scale_n,
            "contact_logit": output[..., 8:10],
            "phase": torch.tanh(output[..., 10:14]).reshape(*output.shape[:-1], 2, 2),
            "sigma_nm": sigma_nm,
            "confidence": 1 / (1 + sigma_nm),
        }

    def forward(self, imu, imu_mask, extra):
        return {name: value[:, -1] for name, value in self.forward_sequence(imu, imu_mask, extra).items()}


def estimator_loss(prediction, target, torque_scale_nm=50.0, grf_scale_n=750.0, loss_config=None):
    options = loss_config or {}
    weights = {"hip": 1., "aux": .3, "grf": .2, "contact": .2, "phase": .2, "nll": .1}
    overrides = options.get("weights", {})
    if set(overrides) - set(weights):
        raise ValueError("Unknown estimator loss weight")
    weights.update(overrides)
    if any(not math.isfinite(value) or value < 0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("Loss weights must be finite, nonnegative and not all zero")
    torque = target["torque"] / torque_scale_nm
    hip = prediction["tau_hip"] / torque_scale_nm
    sigma = prediction["sigma_nm"] / torque_scale_nm
    valid = target["phase_mask"].unsqueeze(-1).expand_as(target["phase"])
    phase_error = (prediction["phase"] - target["phase"]).square()
    phase_loss = (phase_error * valid).sum() / valid.sum().clamp_min(1)
    kind = options.get("hip_type", "huber")
    if kind == "huber":
        hip_loss = F.huber_loss(hip, torque[:, :2])
    elif kind == "smooth_l1":
        beta_nm = float(options.get("hip_beta_nm", 2.))
        if not math.isfinite(beta_nm) or beta_nm <= 0:
            raise ValueError("hip_beta_nm must be positive and finite")
        hip_loss = F.smooth_l1_loss(hip, torque[:, :2], beta=beta_nm / torque_scale_nm)
    elif kind == "l1":
        hip_loss = F.l1_loss(hip, torque[:, :2])
    else:
        raise ValueError(f"Unknown hip loss: {kind}")
    terms = {
        "hip": hip_loss,
        "aux": F.huber_loss(prediction["tau_aux"] / torque_scale_nm, torque[:, 2:]),
        "grf": F.huber_loss(prediction["grf_v"] / grf_scale_n, target["grf_v"] / grf_scale_n),
        "contact": F.binary_cross_entropy_with_logits(prediction["contact_logit"], target["contact"]),
        "phase": phase_loss,
        "nll": F.gaussian_nll_loss(hip, torque[:, :2], sigma.square()),
    }
    loss = sum(weights[name] * value for name, value in terms.items())
    return loss, terms


class RealtimeEstimator:
    """Consume one preprocessed frame at a time, returning physical output units."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.frames = deque(maxlen=model.window_steps)

    def reset(self):
        self.frames.clear()

    @torch.inference_mode()
    def step(self, imu, imu_mask, extra):
        self.frames.append((np.asarray(imu).copy(), np.asarray(imu_mask).copy(), np.asarray(extra).copy()))
        if len(self.frames) < self.model.window_steps:
            return None
        inputs = [torch.as_tensor(np.stack(values)[None], device=self.device, dtype=torch.float32)
                  for values in zip(*self.frames)]
        return {name: value[0].cpu().numpy() for name, value in self.model(*inputs).items()}
