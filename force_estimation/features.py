"""One causal preprocessing path for simulated and physical sensor packets."""

import numpy as np


IMU_NAMES = ("pelvis", "thigh_l", "thigh_r", "shank_l", "shank_r", "foot_l", "foot_r")
JOINT_NAMES = ("hip_flexion_l", "hip_flexion_r", "knee_angle_l", "knee_angle_r",
               "ankle_angle_l", "ankle_angle_r")
EXTRA_NAMES = (
    "hip_angle_l", "hip_angle_r", "hip_velocity_l", "hip_velocity_r",
    "fsr_heel_l", "fsr_toe_l", "fsr_heel_r", "fsr_toe_r",
    "fsr_rate_heel_l", "fsr_rate_toe_l", "fsr_rate_heel_r", "fsr_rate_toe_r",
    "fsr_sum_l", "fsr_sum_r", "toe_ratio_l", "toe_ratio_r",
    "fsr_contact_l", "fsr_contact_r",
    *tuple(f"inclination_{name}" for name in IMU_NAMES[1:]),
    *tuple(f"sagittal_gyro_{name}" for name in IMU_NAMES[1:]),
    "knee_angle_proxy_l", "knee_angle_proxy_r", "ankle_angle_proxy_l", "ankle_angle_proxy_r",
    "hip_acceleration_l", "hip_acceleration_r",
    "fsr_bw_l", "fsr_bw_r", "hip_moment_proxy_l", "hip_moment_proxy_r",
    "hip_sine_l", "hip_sine_r",
    "exo_angle_l", "exo_angle_r", "exo_velocity_l", "exo_velocity_r",
    "exo_delivered_torque_l", "exo_delivered_torque_r", "exo_present_l", "exo_present_r",
    "mass_kg", "height_m", "thigh_length_m", "shank_length_m", "foot_length_m",
    *tuple(f"valid_{name}" for name in IMU_NAMES),
)
HUMAN_EXTRA_NAMES = tuple(name for name in EXTRA_NAMES if not name.startswith("exo_"))


def feature_names(include_exo_features):
    return EXTRA_NAMES if include_exo_features else HUMAN_EXTRA_NAMES


class EncoderFilter:
    def __init__(self, sample_rate_hz=100, cutoff_hz=30):
        self.dt = 1 / float(sample_rate_hz)
        self.coefficient = 1 - np.exp(-2 * np.pi * float(cutoff_hz) * self.dt)
        self.reset()

    def reset(self):
        self.previous = None
        self.velocity = np.zeros(2)

    def step(self, angle):
        angle = np.asarray(angle, dtype=float)
        if angle.shape != (2,) or not np.isfinite(angle).all():
            raise ValueError("Expected two finite hip encoder angles in radians")
        raw_velocity = np.zeros(2) if self.previous is None else (angle - self.previous) / self.dt
        self.previous = angle.copy()
        self.velocity += self.coefficient * (raw_velocity - self.velocity)
        return np.concatenate((angle, self.velocity)).astype(np.float32)


class FeatureBuilder:
    """Only measured streams and known anthropometrics enter these features."""

    def __init__(self, sample_rate_hz, fsr_calibration, subject, include_exo_features=True, fsr_curves=None):
        self.dt = 1.0 / float(sample_rate_hz)
        self.calibration = float(fsr_calibration)
        self.subject = np.asarray(subject, dtype=np.float64)
        if self.subject.shape != (5,) or np.any(self.subject <= 0):
            raise ValueError("subject must contain positive mass, height and three segment lengths")
        self.extra_names = feature_names(include_exo_features)
        self.extra_indices = np.array([EXTRA_NAMES.index(name) for name in self.extra_names])
        self.fsr_curves = None if fsr_curves is None else np.asarray(fsr_curves, dtype=float)
        if self.fsr_curves is not None:
            if (self.fsr_curves.ndim != 3 or self.fsr_curves.shape[0] != 4 or self.fsr_curves.shape[2] != 2
                    or not np.isfinite(self.fsr_curves).all()
                    or np.any(np.diff(self.fsr_curves[:, :, 0], axis=1) <= 0)):
                raise ValueError("FSR calibration must have shape [4,K,2] with increasing readings")
        self.reset()

    def reset(self):
        self.previous_fsr = None
        self.previous_velocity = None
        self.filtered_acceleration = np.zeros(2)

    def step(self, imu, mask, encoder, fsr, exo=None):
        imu = np.asarray(imu, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        encoder = np.asarray(encoder, dtype=np.float64)
        fsr = np.asarray(fsr, dtype=np.float64)
        exo = np.zeros(8) if exo is None else np.asarray(exo, dtype=np.float64)
        if imu.shape != (7, 9) or mask.shape != (7,) or encoder.shape != (4,) or fsr.shape != (4,) or exo.shape != (8,):
            raise ValueError("Expected imu[7,9], mask[7], encoder[4], fsr[4], exo[8]")
        imu = np.where(mask[:, None], imu, 0.0)
        if not all(np.isfinite(value).all() for value in (imu, encoder, fsr, exo)):
            raise ValueError("Active sensor values must be finite")
        fsr_rate = np.zeros(4) if self.previous_fsr is None else (fsr - self.previous_fsr) / self.dt
        self.previous_fsr = fsr.copy()
        acceleration = np.zeros(2) if self.previous_velocity is None else (encoder[2:] - self.previous_velocity) / self.dt
        self.previous_velocity = encoder[2:].copy()
        self.filtered_acceleration += (1 - np.exp(-2 * np.pi * 30 * self.dt)) * (acceleration - self.filtered_acceleration)
        force_channels = fsr * self.calibration
        if self.fsr_curves is not None:
            force_channels = np.array([np.interp(value, curve[:, 0], curve[:, 1])
                                       for value, curve in zip(fsr, self.fsr_curves)])
        positive = np.maximum(fsr if self.fsr_curves is None else force_channels, 0.0).reshape(2, 2)
        foot_sums = positive.sum(axis=1)
        toe_ratios = positive[:, 1] / np.maximum(foot_sums, 1e-6)
        force_proxy = foot_sums * self.calibration if self.fsr_curves is None else foot_sums
        fsr_contacts = (force_proxy > 20.0).astype(float)
        gravity = imu[1:, 6:9]
        inclinations = np.where(mask[1:], np.arctan2(-gravity[:, 0], -gravity[:, 2]), 0.0)
        knee = (inclinations[:2] - inclinations[2:4]) * (mask[1:3] & mask[3:5])
        ankle = (inclinations[2:4] - inclinations[4:6]) * (mask[3:5] & mask[5:7])
        body_weight = self.subject[0] * 9.81
        lever = self.subject[2] * np.sin(inclinations[:2]) + self.subject[3] * np.sin(inclinations[2:4])
        moment_proxy = force_proxy * lever * (mask[1:3] & mask[3:5])
        extra = np.concatenate((encoder, fsr, fsr_rate, foot_sums, toe_ratios, fsr_contacts,
                                inclinations, imu[1:, 1], knee, ankle, self.filtered_acceleration,
                                force_proxy / body_weight, moment_proxy, np.sin(encoder[:2]),
                                exo, self.subject, mask.astype(float))).astype(np.float32)
        if extra.size != len(EXTRA_NAMES) or not np.isfinite(extra).all():
            raise ValueError("Invalid derived features")
        return extra[self.extra_indices]


def contact_phase(contact):
    """Offline stride labels between actual rising contact edges, with edge masks."""
    contact = np.asarray(contact, dtype=bool)
    phase = np.zeros((len(contact), 2, 2), dtype=np.float32)
    valid = np.zeros((len(contact), 2), dtype=bool)
    for side in range(2):
        strikes = np.flatnonzero(contact[1:, side] & ~contact[:-1, side]) + 1
        for start, end in zip(strikes[:-1], strikes[1:]):
            angle = 2 * np.pi * np.arange(end - start) / (end - start)
            phase[start:end, side] = np.stack((np.sin(angle), np.cos(angle)), axis=-1)
            valid[start:end, side] = True
    return phase, valid
