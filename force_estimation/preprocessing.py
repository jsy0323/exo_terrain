"""Causal filtering, six-axis gravity estimation and virtual FSR calibration."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class CausalLowPass:
    def __init__(self, sample_rate_hz, cutoff_hz, order=4):
        if not 0 < cutoff_hz < sample_rate_hz / 2:
            raise ValueError("Filter cutoff must lie between zero and Nyquist")
        self.sos = butter(order, cutoff_hz, fs=sample_rate_hz, output="sos")
        self.state = None

    def reset(self):
        self.state = None

    def step(self, value):
        value = np.asarray(value, dtype=float)
        if self.state is None:
            initial = sosfilt_zi(self.sos)
            self.state = initial.reshape(*initial.shape, *([1] * value.ndim)) * value
        output, self.state = sosfilt(self.sos, value[None], axis=0, zi=self.state)
        return output[0]


class GravityAHRS:
    """Sensor-frame downward gravity, using only gyro and specific force.

    Rotation follows g_dot = -omega x g. Acceleration correction is reduced
    when measured specific-force magnitude differs from the gravity magnitude.
    """

    def __init__(self, sample_rate_hz=100, gain=1.0, startup_s=2.0,
                 acceleration_tolerance_mps2=2.0, rejection_deg=35.0):
        self.dt = 1 / float(sample_rate_hz)
        self.gain = float(gain)
        self.startup_s = float(startup_s)
        self.acceleration_tolerance = float(acceleration_tolerance_mps2)
        self.rejection_rad = np.deg2rad(rejection_deg)
        self.reset()

    def reset(self):
        self.gravity = np.tile([0., 0., -1.], (7, 1))
        self.initialized = np.zeros(7, dtype=bool)
        self.age = 0.0

    def calibrate(self, stationary_acceleration):
        acceleration = np.asarray(stationary_acceleration, dtype=float)
        if acceleration.shape != (7, 3) or not np.isfinite(acceleration).all():
            raise ValueError("Static calibration requires seven finite acceleration vectors")
        magnitude = np.linalg.norm(acceleration, axis=-1, keepdims=True)
        if np.any(magnitude < 1):
            raise ValueError("Static calibration acceleration magnitude is too small")
        self.gravity = -acceleration / magnitude
        self.initialized[:] = True
        self.age = self.startup_s

    def step(self, gyro, acceleration, available):
        gyro, acceleration = np.asarray(gyro), np.asarray(acceleration)
        available = np.asarray(available, dtype=bool)
        magnitude = np.linalg.norm(acceleration, axis=-1)
        measured = -acceleration / np.maximum(magnitude[:, None], 1e-9)
        initialize = available & ~self.initialized & (magnitude > 1)
        self.gravity[initialize] = measured[initialize]
        self.initialized[initialize] = True
        rotation = -gyro * self.dt
        angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
        axis = rotation / np.maximum(angle, 1e-12)
        predicted = (self.gravity * np.cos(angle) + np.cross(axis, self.gravity) * np.sin(angle)
                     + axis * np.sum(axis * self.gravity, axis=-1, keepdims=True) * (1 - np.cos(angle)))
        error_angle = np.arccos(np.clip(np.sum(predicted * measured, axis=-1), -1, 1))
        trust = np.exp(-((magnitude - 9.81) / self.acceleration_tolerance) ** 2)
        startup = self.age < self.startup_s
        if not startup:
            trust *= error_angle < self.rejection_rad
        coefficient = (1 - np.exp(-(5.0 if startup else self.gain) * self.dt)) * trust
        corrected = predicted + coefficient[:, None] * (measured - predicted)
        corrected /= np.maximum(np.linalg.norm(corrected, axis=-1, keepdims=True), 1e-9)
        self.gravity[available] = corrected[available]
        self.age += self.dt
        return self.gravity.copy()


def fsr_calibration_curve(parameters, calibration, rng, noise_std=0.002):
    """Simulate a static calibration with known applied loads and noisy readings."""
    loads = np.array([0., 25., 50., 100., 200., 400., 800., 1200.])
    x = np.maximum(loads[None] - parameters["deadzone"][:, None], 0)
    alpha = parameters["alpha"][:, None]
    readings = parameters["gain"][:, None] * x ** alpha / (1 + (x / parameters["f_sat"][:, None]) ** alpha)
    readings = readings / calibration + parameters["bias"][:, None]
    readings += rng.normal(0, noise_std, readings.shape)
    for index in range(1, readings.shape[1]):
        readings[:, index] = np.maximum(readings[:, index], readings[:, index - 1] + 1e-6)
    return np.stack((readings, np.broadcast_to(loads, readings.shape)), axis=-1)
