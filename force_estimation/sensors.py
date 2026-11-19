"""MuJoCo wearable sites, episode randomization and timestamped acquisition."""

from collections import deque

import mujoco
import numpy as np

from .features import IMU_NAMES, JOINT_NAMES, EncoderFilter, FeatureBuilder, contact_phase
from .preprocessing import CausalLowPass, GravityAHRS, fsr_calibration_curve


BODY_NAMES = ("pelvis", "femur_l", "femur_r", "tibia_l", "tibia_r", "calcn_l", "calcn_r")
SOURCE_SITES = ("pelvis_sensor", "femur_l_sensor", "femur_r_sensor", "tibia_l_sensor",
                "tibia_r_sensor", "calcn_l_sensor", "calcn_r_sensor")


def add_sensor_sites(spec):
    """Called before compilation. These massless sensors leave policy actions intact."""
    for name, body, source in zip(IMU_NAMES, BODY_NAMES, SOURCE_SITES):
        site_name = f"wearable_{name}"
        spec.body(body).add_site(name=site_name, pos=spec.site(source).pos, size=[0.005] * 3, group=5)
        for suffix, kind in (("gyro", mujoco.mjtSensor.mjSENS_GYRO),
                             ("acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
            spec.add_sensor(name=f"{site_name}_{suffix}", type=kind,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)


def fsr_model(force, state, parameters, dt, calibration, control_dt, rng):
    """The specified FSR law, with bias/noise applied in normalized units."""
    x = np.maximum(np.asarray(force) - parameters["deadzone"], 0)
    y = parameters["gain"] * x ** parameters["alpha"] / (1 + (x / parameters["f_sat"]) ** parameters["alpha"])
    y /= calibration
    lag = 1 - (1 - parameters["lag_coef"]) ** (dt / control_dt)
    state["y"] += (y - state["y"]) * lag
    state["drift"] += parameters["creep"] * y * dt
    return state["y"] + state["drift"] + parameters["bias"] + parameters["sigma"] * rng.normal(size=4)


def inverse_dynamics_moment(model, data, dof_indices):
    """Net non-constraint joint demand, including active and passive contributions.

    M(q) qacc + qfrc_bias - qfrc_constraint. This differs from qfrc_inverse,
    which also subtracts passive forces in MuJoCo's convention.
    """
    inertial = np.empty(model.nv)
    mujoco.mj_mulM(model, data, inertial, data.qacc)
    return (inertial + data.qfrc_bias - data.qfrc_constraint)[dof_indices].copy()


class WearableSimulator:
    def __init__(self, env, config):
        self.env = env
        self.model = env.model
        self.config = config
        self.rate = float(config["sample_rate_hz"])
        self.dt = 1 / self.rate
        physics_dt = float(self.model.opt.timestep)
        ratio = self.dt / physics_dt
        if abs(ratio - round(ratio)) > 1e-8 or ratio < 1:
            raise ValueError("Wearable sample period must be an integer number of physics steps")
        self.sample_every = int(round(ratio))
        self.processing = config.get("preprocessing", {})
        cutoff = self.processing.get("anti_alias_hz")
        self.imu_lowpass = CausalLowPass(1 / physics_dt, cutoff) if cutoff else None
        self.label_lowpass = CausalLowPass(1 / physics_dt, cutoff) if cutoff else None
        self.ahrs = None
        if self.processing.get("gravity_source") == "ahrs":
            self.ahrs = GravityAHRS(self.rate, **self.processing.get("ahrs", {}))
        self.sample_callback = None
        self.calibration = float(config["fsr"]["calibration_value"])
        if self.calibration <= 0:
            raise ValueError("FSR calibration_value must be positive")
        self.site_ids = np.array([self.model.site(f"wearable_{name}").id for name in IMU_NAMES])
        self.gyro_addresses = [int(self.model.sensor(f"wearable_{name}_gyro").adr[0]) for name in IMU_NAMES]
        self.acc_addresses = [int(self.model.sensor(f"wearable_{name}_acc").adr[0]) for name in IMU_NAMES]
        self.nominal_positions = self.model.site_pos[self.site_ids].copy()
        # Calibrated neutral frames have +X forward, +Y left, +Z up.
        neutral = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, neutral)
        self.nominal_quaternions = []
        for body in BODY_NAMES:
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, neutral.xmat[self.model.body(body).id].reshape(3, 3).T.ravel())
            self.nominal_quaternions.append(quaternion)
        self.nominal_quaternions = np.array(self.nominal_quaternions)
        self.dof_indices = np.array([int(self.model.joint(name).dofadr[0]) for name in JOINT_NAMES])
        self.hip_qpos = np.array([int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES[:2]])
        self.foot_geoms = []
        self.region_boundaries = []
        for side, foot in (("l", "calcn_l"), ("r", "calcn_r")):
            geoms = [self.model.geom(name).id for name in (f"C_{side}_foot", f"C_{side}_bofoot1", f"C_{side}_bofoot2")]
            self.foot_geoms.append(set(geoms))
            body_id = self.model.body(foot).id
            origin = neutral.xpos[body_id]
            # Neutral calibrated forward coordinate equals world X.
            forward = neutral.geom_xpos[geoms, 0] - origin[0]
            self.region_boundaries.append(float((forward[0] + np.mean(forward[1:])) / 2))
        # Use explicit ordering even if YAML order is changed.
        subject = config["subject"]
        self.subject = np.array([np.sum(self.model.body_mass), subject["height_m"], subject["thigh_length_m"],
                                 subject["shank_length_m"], subject["foot_length_m"]], dtype=float)
        self.features = FeatureBuilder(
            self.rate, self.calibration, self.subject,
            include_exo_features=bool(config["training"].get("include_exo_features", False)),
        )
        self.encoder_filter = EncoderFilter(self.rate, config["encoder"]["cutoff_hz"])
        max_delay = float(config["imu"]["delay_ms"][1]) / 1000
        self.history = deque(maxlen=int(np.ceil(max_delay / physics_dt)) + 3)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        imu = self.config["imu"]
        self.position_offsets = self.rng.uniform(-imu["position_offset_m"], imu["position_offset_m"], (7, 3))
        angles = np.deg2rad(self.rng.uniform(-imu["angle_offset_deg"], imu["angle_offset_deg"], (7, 3)))
        self.model.site_pos[self.site_ids] = self.nominal_positions + self.position_offsets
        for index, site_id in enumerate(self.site_ids):
            delta = np.empty(4)
            mujoco.mju_euler2Quat(delta, angles[index], "xyz")
            quaternion = np.empty(4)
            mujoco.mju_mulQuat(quaternion, self.nominal_quaternions[index], delta)
            self.model.site_quat[site_id] = quaternion
        self.gyro_bias = self.rng.uniform(-imu["gyro_bias"], imu["gyro_bias"], (7, 3))
        self.acc_bias = self.rng.uniform(-imu["acc_bias"], imu["acc_bias"], (7, 3))
        self.vibration_frequency = self.rng.uniform(*imu["vibration_hz"], (7, 3))
        self.vibration_amplitude = self.rng.uniform(0, imu["vibration_max_mps2"], (7, 3))
        self.vibration_phase = self.rng.uniform(0, 2 * np.pi, (7, 3))
        self.delays = self.rng.uniform(*imu["delay_ms"], 7) / 1000
        self.imu_mask = self.rng.random(7) >= float(imu.get("dropout_probability", 0.0))
        fsr = self.config["fsr"]
        self.fsr_parameters = {name: self.rng.uniform(*fsr[name], 4)
                               for name in ("deadzone", "gain", "alpha", "f_sat", "lag_coef", "creep", "bias")}
        self.fsr_parameters["sigma"] = float(fsr["sigma"])
        self.calibration_curves = None
        if self.processing.get("fsr_calibration") == "bench":
            self.calibration_curves = fsr_calibration_curve(
                self.fsr_parameters, self.calibration, self.rng,
                noise_std=float(self.processing.get("calibration_noise_std", .002)),
            )
        self.features.fsr_curves = self.calibration_curves
        self.boundary_offsets = self.rng.uniform(-fsr["boundary_offset_m"], fsr["boundary_offset_m"], 2)
        self.fsr_state = {"y": np.zeros(4), "drift": np.zeros(4)}
        self.encoder_filter.reset()
        self.features.reset()
        self.history.clear()
        self.frames = []
        self.tick = 0
        self.start_time = float(self.env.data.time)
        self.subject[0] = np.sum(self.model.body_mass)
        self.features.subject[:] = self.subject
        self.dropout_until = np.zeros(7)
        if self.imu_lowpass is not None:
            self.imu_lowpass.reset()
            self.label_lowpass.reset()
        if self.ahrs is not None:
            self.ahrs.reset()
        mujoco.mj_forward(self.model, self.env.data)
        self.static_acceleration_calibration = None
        if self.ahrs is not None:
            # A held-pose calibration provides an accelerometer measurement.
            # Subsequent AHRS updates use only the live noisy six-axis stream.
            static_gravity = self._ideal_imu(self.env.data)[:, 6:9]
            self.static_acceleration_calibration = (-static_gravity * 9.81 + self.acc_bias
                                                    + self.rng.normal(0, imu["acc_noise_std"] / np.sqrt(100), (7, 3)))
            self.ahrs.calibrate(self.static_acceleration_calibration)
        self.history.append((self.start_time, self._ideal_imu(self.env.data)))

    def _ideal_imu(self, data):
        result = np.empty((7, 9))
        gravity = self.model.opt.gravity / np.linalg.norm(self.model.opt.gravity)
        for index, site_id in enumerate(self.site_ids):
            result[index, :3] = data.sensordata[self.gyro_addresses[index]:self.gyro_addresses[index] + 3]
            result[index, 3:6] = data.sensordata[self.acc_addresses[index]:self.acc_addresses[index] + 3]
            result[index, 6:9] = data.site_xmat[site_id].reshape(3, 3).T @ gravity
        return result

    def _foot_forces(self, data):
        region = np.zeros((2, 2))
        grf = np.zeros(2)
        force = np.empty(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            for side, geoms in enumerate(self.foot_geoms):
                first, second = int(contact.geom1) in geoms, int(contact.geom2) in geoms
                if first == second:
                    continue
                other = int(contact.geom2 if first else contact.geom1)
                if int(self.model.geom_bodyid[other]) != 0:
                    continue
                mujoco.mj_contactForce(self.model, data, contact_id, force)
                world_force = contact.frame.reshape(3, 3).T @ force[:3]
                grf[side] += (-1 if first else 1) * world_force[2]
                site_id = self.site_ids[5 + side]
                # Use the calibrated anatomical frame, excluding mounting errors.
                body_id = int(self.model.site_bodyid[site_id])
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, self.nominal_quaternions[5 + side])
                rotation = data.xmat[body_id].reshape(3, 3) @ rotation.reshape(3, 3)
                forward_position = float((rotation.T @ (contact.pos - data.xpos[body_id]))[0])
                toe = forward_position >= self.region_boundaries[side] + self.boundary_offsets[side]
                region[side, int(toe)] += max(0.0, float(force[0]))
        return region.ravel(), np.maximum(grf, 0)

    def on_physics_step(self, model, data):
        self.tick += 1
        ideal = self._ideal_imu(data)
        filtered = ideal.copy()
        if self.imu_lowpass is not None:
            filtered[:, :6] = self.imu_lowpass.step(ideal[:, :6])
            raw_force, raw_grf = self._foot_forces(data)
            raw_torque = inverse_dynamics_moment(model, data, self.dof_indices)
            raw_applied = data.qfrc_applied[self.dof_indices].copy()
            labels = self.label_lowpass.step(np.concatenate((raw_torque, raw_applied, raw_grf, raw_force)))
        self.history.append((float(data.time), filtered))
        if self.tick % self.sample_every:
            return
        times = np.array([item[0] for item in self.history])
        history = np.stack([item[1] for item in self.history])
        delayed = np.empty((7, 9))
        for index, delay in enumerate(self.delays):
            timestamp = float(data.time) - delay
            upper = int(np.clip(np.searchsorted(times, timestamp, side="right"), 1, len(times) - 1))
            fraction = np.clip((timestamp - times[upper - 1]) / (times[upper] - times[upper - 1]), 0, 1)
            delayed[index] = history[upper - 1, index] * (1 - fraction) + history[upper, index] * fraction
        delayed[:, 6:9] /= np.maximum(np.linalg.norm(delayed[:, 6:9], axis=-1, keepdims=True), 1e-9)
        imu = delayed.copy()
        imu_cfg = self.config["imu"]
        imu[:, :3] += self.gyro_bias + self.rng.normal(0, imu_cfg["gyro_noise_std"], (7, 3))
        vibration_time = float(data.time) - self.delays[:, None]
        vibration = self.vibration_amplitude * np.sin(2 * np.pi * self.vibration_frequency * vibration_time + self.vibration_phase)
        imu[:, 3:6] += self.acc_bias + self.rng.normal(0, imu_cfg["acc_noise_std"], (7, 3)) + vibration
        elapsed = float(data.time) - self.start_time
        burst_rate = float(imu_cfg.get("dropout_burst_rate_hz", 0))
        if burst_rate > 0:
            new_burst = self.rng.random(7) < 1 - np.exp(-burst_rate * self.dt)
            duration = self.rng.uniform(*imu_cfg.get("dropout_burst_s", [.1, .4]), 7)
            self.dropout_until[new_burst] = elapsed + duration[new_burst]
        available = self.imu_mask & (elapsed >= self.dropout_until)
        if self.ahrs is not None:
            imu[:, 6:9] = self.ahrs.step(imu[:, :3], imu[:, 3:6], available)
        imu[~available] = 0
        encoder_cfg = self.config["encoder"]
        angle = data.qpos[self.hip_qpos] + self.rng.normal(0, encoder_cfg["noise_std_rad"], 2)
        angle = np.round(angle / encoder_cfg["quantization_rad"]) * encoder_cfg["quantization_rad"]
        encoder = self.encoder_filter.step(angle)
        if self.label_lowpass is None:
            raw_force, raw_grf = self._foot_forces(data)
            raw_torque = inverse_dynamics_moment(model, data, self.dof_indices)
            raw_applied = data.qfrc_applied[self.dof_indices].copy()
            force, grf, torque, applied = raw_force, raw_grf, raw_torque, raw_applied
        else:
            torque, applied = labels[:6], labels[6:12]
            grf, force = np.maximum(labels[12:14], 0), np.maximum(labels[14:18], 0)
        fsr = fsr_model(force, self.fsr_state, self.fsr_parameters, self.dt, self.calibration,
                        self.config["fsr"]["control_dt_s"], self.rng)
        exo = np.zeros(8)
        extra = self.features.step(imu, available, encoder, fsr, exo)
        contact = (raw_grf >= float(self.config["contact_threshold_n"])).astype(np.float32)
        command = self.env.speed_command
        direction = self.env.speed_command_direction[:2]
        frame = {"time": elapsed, "imu": imu, "imu_ideal": ideal,
                 "imu_mask": available.copy(), "encoder": encoder, "fsr": fsr,
                 "fsr_force_n": force, "fsr_force_raw_n": raw_force, "extra": extra,
                 "tau_total": torque.copy(), "tau_bio": (torque - applied).copy(),
                 "tau_exo": applied.copy(), "tau_applied": applied.copy(),
                 "tau_total_raw": raw_torque.copy(), "tau_bio_raw": raw_torque - raw_applied,
                 "tau_applied_raw": raw_applied.copy(), "grf_v": grf, "grf_v_raw": raw_grf,
                 "contact": contact, "pelvis_velocity": data.qvel[:3].copy(),
                 "pelvis_height": float(data.xpos[model.body('pelvis').id, 2]),
                 "forward_speed": float(np.dot(data.qvel[:2], direction)),
                 "command_speed": float(command.speed if command is not None else self.env.ref_traj.speed),
                 "sample_valid": elapsed >= float(self.processing.get("warmup_s", 0))}
        self.frames.append(frame)
        if self.sample_callback is not None:
            self.sample_callback(frame)

    def arrays(self):
        if not self.frames:
            raise ValueError("Episode has no wearable samples")
        arrays = {name: np.asarray([frame[name] for frame in self.frames], dtype=np.float32) for name in self.frames[0]}
        arrays["time"] = np.asarray([frame["time"] for frame in self.frames], dtype=np.float64)
        arrays["imu_mask"] = arrays["imu_mask"].astype(bool)
        arrays["sample_valid"] = arrays["sample_valid"].astype(bool)
        arrays["phase"], arrays["phase_mask"] = contact_phase(arrays["contact"])
        return arrays

    def randomization_metadata(self):
        return {"site_ids": self.site_ids.tolist(), "position_offsets_m": self.position_offsets.tolist(),
                "mounting_quaternions": self.model.site_quat[self.site_ids].tolist(),
                "gyro_bias": self.gyro_bias.tolist(), "acc_bias": self.acc_bias.tolist(),
                "delay_s": self.delays.tolist(), "vibration_frequency_hz": self.vibration_frequency.tolist(),
                "vibration_amplitude": self.vibration_amplitude.tolist(), "vibration_phase": self.vibration_phase.tolist(),
                "boundary_offsets_m": self.boundary_offsets.tolist(), "fsr_parameters": {
                    key: np.asarray(value).tolist() for key, value in self.fsr_parameters.items()},
                "fsr_calibration_curves": None if self.calibration_curves is None else self.calibration_curves.tolist(),
                "static_acceleration_calibration": None if self.static_acceleration_calibration is None else self.static_acceleration_calibration.tolist()}
