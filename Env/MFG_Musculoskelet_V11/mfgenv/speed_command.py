"""A walking-speed hold followed by a constant-acceleration ramp and run hold."""

import math

import numpy as np


class SpeedCommand:
    def __init__(self, config):
        self.start_speed = float(config.get("start_speed", 1.4))
        self.end_speed = float(config.get("end_speed", 2.8))
        self.hold_seconds = float(config.get("hold_seconds", 2.0))
        self.acceleration_range = tuple(config.get("acceleration_range", (0.3, 1.0)))
        if len(self.acceleration_range) != 2:
            raise ValueError("speed_command.acceleration_range needs two values")
        self.acceleration_range = tuple(float(v) for v in self.acceleration_range)
        values = (self.start_speed, self.end_speed, self.hold_seconds, *self.acceleration_range)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("speed_command values must be finite")
        if min(self.start_speed, self.end_speed) < 0 or self.hold_seconds < 0:
            raise ValueError("speed_command requires nonnegative speeds and hold_seconds")
        if not 0 < self.acceleration_range[0] <= self.acceleration_range[1]:
            raise ValueError("speed_command requires 0 < min_acceleration <= max_acceleration")
        self.ramp_acceleration = self.acceleration_range[0]
        self.speed = self.start_speed
        self.acceleration = 0.0

    def reset(self, rng):
        self.ramp_acceleration = float(rng.uniform(*self.acceleration_range))
        self.update(0.0)

    def update(self, elapsed_seconds):
        ramp_time = max(0.0, float(elapsed_seconds) - self.hold_seconds)
        direction = float(np.sign(self.end_speed - self.start_speed))
        distance = min(abs(self.end_speed - self.start_speed), self.ramp_acceleration * ramp_time)
        self.speed = self.start_speed + direction * distance
        self.acceleration = 0.0
        if elapsed_seconds >= self.hold_seconds and distance < abs(self.end_speed - self.start_speed):
            self.acceleration = direction * self.ramp_acceleration

    def observation(self):
        return np.array((self.speed, self.acceleration), dtype=np.float32)
