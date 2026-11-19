"""Physical mass profiles, motion schedules and ideal joint-torque sources."""

import numpy as np
import mujoco

from Env.MFG_Musculoskelet_V11.mfgenv.speed_command import SpeedCommand


class SimulationConditions:
    def __init__(self, env, config):
        self.env = env
        self.options = config.get("experiments", {})
        self.mass = env.model.body_mass.copy()
        self.inertia = env.model.body_inertia.copy()
        self.subjects = self.options.get("subjects", [{"id": "nominal", "mass_scale": 1.0}])
        self.motions = self.options.get("motions", [{"id": "original"}])
        self.repeats = int(self.options.get("repeats", 1))
        self.closed_loop = None
        self.desired_hip_torque = np.zeros(2)
        self.applied_hip_torque = np.zeros(2)
        self.hip_dofs = np.array([int(env.model.joint(f"hip_flexion_{side}").dofadr[0]) for side in ("l", "r")])
        self.motion = {}

    def select(self, episode):
        motion_index = (episode // self.repeats) % len(self.motions)
        subject_index = (episode // (self.repeats * len(self.motions))) % len(self.subjects)
        self.subject = self.subjects[subject_index]
        self.motion = self.motions[motion_index]
        scale = float(self.subject["mass_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("mass_scale must be finite and positive")
        self.env.model.body_mass[:] = self.mass * scale
        self.env.model.body_inertia[:] = self.inertia * scale
        mujoco.mj_setConst(self.env.model, mujoco.MjData(self.env.model))
        self.env.total_mass = float(np.sum(self.env.model.body_mass))
        if "start_speed" in self.motion:
            self.env.speed_command = SpeedCommand(self.motion)
        self.applied_hip_torque[:] = 0
        self.desired_hip_torque[:] = 0
        return {"subject_id": self.subject["id"], "motion_id": self.motion["id"],
                "session_id": f"{self.subject['id']}/{self.motion['id']}/{episode % self.repeats}",
                "mass_scale": scale, "geometry_scale": 1.0}

    def apply_torque(self, model, data):
        if self.closed_loop is None:
            amplitude = float(self.motion.get("hip_torque_amplitude_nm", 0))
            frequency = float(self.motion.get("hip_torque_frequency_hz", 0.8))
            self.desired_hip_torque = amplitude * np.sin(2 * np.pi * frequency * data.time + np.array([0, np.pi]))
        limit = float(self.options.get("torque_limit_nm", 4.0))
        slew = float(self.options.get("torque_slew_nm_s", 40.0)) * model.opt.timestep
        desired = np.clip(self.desired_hip_torque, -limit, limit)
        self.applied_hip_torque += np.clip(desired - self.applied_hip_torque, -slew, slew)
        data.qfrc_applied[self.hip_dofs] = self.applied_hip_torque

    def restore(self):
        self.env.model.body_mass[:] = self.mass
        self.env.model.body_inertia[:] = self.inertia
        self.env.data.qfrc_applied[self.hip_dofs] = 0
        mujoco.mj_setConst(self.env.model, mujoco.MjData(self.env.model))
        self.env.total_mass = float(np.sum(self.mass))


class EstimatorFeedback:
    def __init__(self, checkpoint, simulator, conditions, gain):
        from .runtime import OnlineMomentEstimator
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("Assistance gain must be finite and nonnegative")
        self.estimator = OnlineMomentEstimator(checkpoint, simulator.subject, device="cpu",
                                               fsr_calibration_curves=simulator.calibration_curves)
        self.conditions = conditions
        self.gain = float(gain)
        self.ready_time = float(simulator.processing.get("warmup_s", 0)) + self.estimator.model.window_steps / simulator.rate
        conditions.closed_loop = self

    def on_sample(self, frame):
        prediction = self.estimator.step(frame["time"], frame["imu"], frame["imu_mask"],
                                          frame["encoder"][:2], frame["fsr"])
        frame["estimated_tau_total_hip"] = np.zeros(2)
        frame["estimated_tau_bio_hip"] = np.zeros(2)
        frame["estimated_sigma_nm"] = np.zeros(2)
        frame["requested_hip_torque"] = np.zeros(2)
        frame["estimator_ready"] = prediction is not None and frame["time"] >= self.ready_time
        if prediction is None:
            return
        applied = frame["tau_applied"][:2]
        biological = prediction["tau_hip"] - applied if self.estimator.target == "total" else prediction["tau_hip"]
        if not np.isfinite(biological).all():
            self.conditions.desired_hip_torque[:] = 0
            raise FloatingPointError("Non-finite estimator feedback")
        frame["estimated_tau_bio_hip"] = biological.copy()
        frame["estimated_tau_total_hip"] = biological + applied
        frame["estimated_sigma_nm"] = prediction["sigma_nm"].copy()
        if frame["estimator_ready"]:
            self.conditions.desired_hip_torque = self.gain * biological
            frame["requested_hip_torque"] = self.conditions.desired_hip_torque.copy()
