"""Stateful 100 Hz inference using the same causal features as collection."""

import numpy as np
import torch

from .features import EXTRA_NAMES, HUMAN_EXTRA_NAMES, EncoderFilter, FeatureBuilder
from .model import MomentEstimator, RealtimeEstimator


class OnlineMomentEstimator:
    def __init__(self, checkpoint_path, subject, device="cpu", fsr_calibration_curves=None):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["extra_names"] not in (list(EXTRA_NAMES), list(HUMAN_EXTRA_NAMES)):
            raise ValueError("Estimator checkpoint feature schema mismatch")
        self.include_exo_features = checkpoint["extra_names"] == list(EXTRA_NAMES)
        self.target = checkpoint["target"]
        self.config = checkpoint["config"]
        self.model = MomentEstimator(**checkpoint["model_args"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.uncertainty_scale.copy_(torch.as_tensor(checkpoint.get("uncertainty_scale", [1., 1.])))
        self.stream = RealtimeEstimator(self.model, device)
        self.encoder = EncoderFilter(self.config["sample_rate_hz"], self.config["encoder"]["cutoff_hz"])
        if self.config.get("preprocessing", {}).get("fsr_calibration") == "bench" and fsr_calibration_curves is None:
            raise ValueError("This checkpoint requires the measured FSR calibration curves")
        self.features = FeatureBuilder(
            self.config["sample_rate_hz"], self.config["fsr"]["calibration_value"], subject,
            include_exo_features=self.include_exo_features,
            fsr_curves=fsr_calibration_curves,
        )
        self.previous_timestamp = None

    def reset(self):
        self.encoder.reset()
        self.features.reset()
        self.stream.reset()
        self.previous_timestamp = None

    def step(self, timestamp, imu, imu_mask, hip_angles, fsr, exo=None):
        """IMU order is pelvis,L/R thigh,L/R shank,L/R foot, with sensor-local vectors.

        IMU fields are gyro[rad/s], specific force[m/s^2], unit gravity direction.
        Hardware gravity direction must come from a causal orientation estimator.
        FSR values use the same calibration_value as the training data.
        exo is [angles(2), rates(2), delivered torques(2), availability(2)].
        """
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        dt = 1 / float(self.config["sample_rate_hz"])
        if self.previous_timestamp is not None and not np.isclose(timestamp - self.previous_timestamp, dt, atol=dt * .05):
            raise ValueError("Resample sensors to the configured uniform rate and reset after discontinuities")
        imu = np.asarray(imu, dtype=np.float32)
        imu_mask = np.asarray(imu_mask, dtype=bool)
        hip_angles = np.asarray(hip_angles, dtype=float)
        fsr = np.asarray(fsr, dtype=float)
        exo = np.zeros(8) if exo is None else np.asarray(exo, dtype=float)
        if imu.shape != (7, 9) or imu_mask.shape != (7,) or hip_angles.shape != (2,) or fsr.shape != (4,) or exo.shape != (8,):
            raise ValueError("Invalid wearable packet dimensions")
        imu = np.where(imu_mask[:, None], imu, 0.0)
        if not all(np.isfinite(value).all() for value in (imu, hip_angles, fsr, exo)):
            raise ValueError("Active wearable measurements must be finite")
        if not self.include_exo_features and np.any(exo != 0):
            raise ValueError("This checkpoint uses human-only inputs. Powered operation requires a compatible trained model.")
        extra = self.features.step(imu, imu_mask, self.encoder.step(hip_angles), fsr, exo)
        self.previous_timestamp = timestamp
        return self.stream.step(imu, imu_mask, extra)
