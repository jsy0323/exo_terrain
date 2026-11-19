#!/usr/bin/env python3
"""Export a cable-policy gait cycle, bounded assistance, and its Fourier series.

The exported curve is an offline, periodic reference. Scaling/filtering does not
modify the checkpoint or the controls applied during the recorded rollout.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def write_csv(path, columns, values):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(values)


def evaluate_series(coefficients, phase):
    """Evaluate a real Fourier series, with phase in cycles (0..1)."""
    angles = 2 * np.pi * np.outer(np.asarray(phase), np.arange(len(coefficients)))
    return np.cos(angles) @ coefficients.real - np.sin(angles) @ coefficients.imag


def bandlimit_assistance(forces, dt, cutoff_hz=8.0, max_force=100.0):
    """Hard FFT cutoff, then affine bounds without nonlinear clipping.

    A dense extrema grid plus a derivative bound encloses the continuous Fourier
    curve, so the 0..max_force bound also holds between exported sample points.
    Both legs share one gain, while any negative ringing is removed by a DC
    offset per leg. Neither operation introduces higher frequency harmonics.
    """
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 2 or len(forces) < 4:
        raise ValueError("Expected at least four samples of right/left cable force.")
    if not np.isfinite(forces).all() or np.min(forces) < -1e-8:
        raise ValueError("Cable tensions must be finite and nonnegative.")
    if (not np.isfinite([dt, cutoff_hz, max_force]).all()
            or dt <= 0 or not 0 < cutoff_hz < 0.5 / dt or max_force <= 0):
        raise ValueError("Require dt > 0, 0 < cutoff < Nyquist, max_force > 0.")
    peak = float(np.max(forces))
    if peak <= 1e-12:
        raise ValueError("The selected cycle contains no assistance force.")
    raw_gain = min(1.0, max_force / peak)
    scaled = forces * raw_gain
    n = len(forces)
    frequency = np.fft.rfftfreq(n, dt)
    # c[0] + sum(real(c[k])*cos(k*theta) - imag(c[k])*sin(k*theta)).
    coefficients = np.fft.rfft(scaled, axis=0) * (2.0 / n)
    coefficients[0] *= 0.5
    if n % 2 == 0:
        coefficients[-1] *= 0.5
    retained = frequency <= cutoff_hz + 1e-12
    filtered = coefficients.copy()
    filtered[~retained] = 0
    grid_size = 16384
    dense = evaluate_series(filtered, np.arange(grid_size) / grid_size)
    # In units of force per cycle: |f'| <= 2*pi*sum(k*|c[k]|).
    margin = (np.pi / grid_size) * np.sum(
        np.arange(len(filtered))[:, None] * np.abs(filtered), axis=0
    )
    lower = dense.min(axis=0) - margin
    offset = np.maximum(0, -lower)
    filtered[0] += offset
    upper = float(np.max(dense.max(axis=0) + margin + offset))
    final_gain = min(1.0, max_force / upper) if upper > 0 else 1.0
    filtered *= final_gain
    filtered_samples = evaluate_series(filtered, np.arange(n) / n)
    high = frequency > cutoff_hz + 1e-12
    reconstructed_fft = np.fft.rfft(filtered_samples, axis=0)
    ac_energy = np.sum(np.abs(reconstructed_fft[1:]) ** 2, axis=0)
    high_energy = np.sum(np.abs(reconstructed_fft[high]) ** 2, axis=0)
    return {
        "scaled": scaled,
        "coefficients_raw": coefficients,
        "coefficients": filtered,
        "frequency": frequency,
        "retained": retained,
        "samples": filtered_samples,
        "raw_gain": raw_gain,
        "dc_offset_n": offset.tolist(),
        "final_gain": final_gain,
        "continuous_lower_bound_n": ((lower + offset) * final_gain).tolist(),
        "continuous_upper_bound_n": (
            (dense.max(axis=0) + margin + offset) * final_gain
        ).tolist(),
        "high_frequency_ac_energy_fraction": np.divide(
            high_energy, ac_energy, out=np.zeros(2), where=ac_energy > 0
        ).tolist(),
    }


def fit_assistance_quiet_phase(
    force, dt, quiet_end_percent, peak_percent, cutoff_hz=8.0,
    max_force=100.0, quiet_limit_n=0.5, fall_end_percent=None,
    peak_force_n=None, tail_start_percent=None, tail_limit_n=0.5,
):
    """Fit a bandlimited reference with suppressed pre-rise force.

    This is constrained waveform fitting, rather than an unmodified model
    output or a causal filter. The quiet interval and peak timing are explicit
    assumptions. Optional peak_force_n fixes the amplitude at peak_percent;
    tail_start_percent bounds the final interval by tail_limit_n.
    Only Fourier harmonics <= cutoff_hz can be produced.
    """
    from scipy.optimize import LinearConstraint, linprog, minimize

    force = np.asarray(force, dtype=float)
    values = [dt, quiet_end_percent, peak_percent, cutoff_hz, max_force, quiet_limit_n]
    if (force.ndim != 1 or len(force) < 8 or not np.isfinite(force).all()
            or not np.isfinite(values).all() or np.min(force) < 0
            or np.max(force) <= 0 or dt <= 0
            or not 0 < cutoff_hz < .5 / dt or max_force <= 0
            or not 0 < quiet_end_percent < peak_percent < 100
            or not .01 < quiet_limit_n < max_force):
        raise ValueError("Invalid force, sampling, force bounds, or phase intervals.")
    n = len(force)
    period = n * dt
    harmonic_count = int(np.floor(cutoff_hz * period + 1e-12))
    if harmonic_count < 2:
        raise ValueError("Too few harmonics for a quiet interval and a distinct peak.")
    k = np.arange(1, harmonic_count + 1)
    quiet_end = quiet_end_percent / 100
    peak_phase = peak_percent / 100
    if fall_end_percent is not None:
        if not np.isfinite(fall_end_percent) or not peak_percent < fall_end_percent < 100:
            raise ValueError("fall_end_percent must be between peak_percent and 100.")
    if peak_force_n is not None:
        if not np.isfinite(peak_force_n) or not quiet_limit_n < peak_force_n <= max_force:
            raise ValueError("peak_force_n must exceed quiet_limit_n and not exceed max_force.")
    if tail_start_percent is not None:
        if (not np.isfinite([tail_start_percent, tail_limit_n]).all()
                or not peak_percent < tail_start_percent < 100
                or not .01 < tail_limit_n < max_force):
            raise ValueError("Invalid tail phase or tail force limit.")

    def basis(phase, derivative=False):
        angles = 2 * np.pi * np.outer(phase, k)
        if derivative:
            return np.column_stack((np.zeros(len(phase)),
                -np.sin(angles) * (2 * np.pi * k),
                np.cos(angles) * (2 * np.pi * k)))
        return np.column_stack((np.ones(len(phase)), np.cos(angles), np.sin(angles)))

    raw_gain = min(1.0, max_force / float(force.max()))
    scaled = force * raw_gain
    sample_phase = np.arange(n) / n
    design = basis(sample_phase)
    weights = np.where(sample_phase <= quiet_end, 8.0, 1.0)
    penalty = 1e-5 * np.r_[0.0, k.astype(float)**4, k.astype(float)**4]
    quadratic = design.T @ (weights[:, None] * design) / n + np.diag(penalty)
    target = design.T @ (weights * scaled / max_force) / n
    grid = np.unique(np.r_[np.arange(2048) / 2048, quiet_end, peak_phase])
    quiet_grid = np.linspace(0, quiet_end, 400)
    # Small margins protect the continuous curve between constraint samples.
    floor = min(2e-5, quiet_limit_n / max_force / 100)
    upper = 1.0 if peak_force_n is not None else .999
    constraints = [
        LinearConstraint(basis(grid), floor, upper),
        LinearConstraint(basis(quiet_grid), floor, .99 * quiet_limit_n / max_force),
        LinearConstraint(basis(np.linspace(quiet_end, peak_phase, 200), True), 0, np.inf),
        LinearConstraint(basis(np.array([peak_phase]), True), 0, 0),
    ]
    if peak_force_n is not None:
        target_peak = peak_force_n / max_force
        constraints.append(LinearConstraint(basis(np.array([peak_phase])), target_peak, target_peak))
    if fall_end_percent is not None:
        constraints.append(LinearConstraint(
            basis(np.linspace(peak_phase, fall_end_percent / 100, 3000), True), -np.inf, 0
        ))
    if tail_start_percent is not None:
        constraints.append(LinearConstraint(
            basis(np.linspace(tail_start_percent / 100, 1, 600)),
            floor, .99 * tail_limit_n / max_force,
        ))
    # Start from the unconstrained fit. A constant start activates every
    # derivative constraint at once and can stall SLSQP at an almost flat fit.
    initial = np.linalg.solve(quadratic, target)
    lift = lambda x: x
    gradient_map = np.eye(len(initial))
    optimize_constraints = constraints
    if peak_force_n is not None:
        # Eliminate the two peak equalities analytically. In particular, the
        # force upper bound at the peak then becomes a constant constraint;
        # retaining its nearly zero numerical gradient destabilizes SLSQP.
        equal_matrix = np.vstack((basis(np.array([peak_phase])),
                                  basis(np.array([peak_phase]), True)))
        equal_target = np.array([peak_force_n / max_force, 0.0])
        particular = np.linalg.lstsq(equal_matrix, equal_target, rcond=None)[0]
        _, _, vh = np.linalg.svd(equal_matrix, full_matrices=True)
        gradient_map = vh[2:].T
        lift = lambda x: particular + gradient_map @ x
        initial = gradient_map.T @ (initial - particular)
        optimize_constraints = []
        for constraint in constraints:
            matrix = np.atleast_2d(constraint.A)
            shift = matrix @ particular
            reduced = matrix @ gradient_map
            lower = np.broadcast_to(constraint.lb, len(matrix)) - shift
            upper_bound = np.broadcast_to(constraint.ub, len(matrix)) - shift
            keep = np.linalg.norm(reduced, axis=1) > 1e-10
            if np.any(lower[~keep] > 1e-9) or np.any(upper_bound[~keep] < -1e-9):
                raise RuntimeError("Peak equalities conflict with a force or slope constraint.")
            if np.any(keep):
                optimize_constraints.append(LinearConstraint(
                    reduced[keep], lower[keep], upper_bound[keep]))

    def objective(x):
        c = lift(x)
        return float(c @ quadratic @ c - 2 * target @ c)

    def objective_gradient(x):
        return gradient_map.T @ (2 * (quadratic @ lift(x) - target))

    solution = minimize(
        objective, initial, jac=objective_gradient,
        constraints=optimize_constraints, method="SLSQP",
        options={"maxiter": 1500, "ftol": 1e-12},
    )
    if not solution.success and peak_force_n is not None:
        # An exact peak on the upper bound can defeat SLSQP's initial line
        # search. Establish feasibility independently, then optimize from it.
        upper_rows, upper_values, equal_rows, equal_values = [], [], [], []
        for constraint in optimize_constraints:
            matrix = np.atleast_2d(constraint.A)
            lows = np.broadcast_to(constraint.lb, len(matrix))
            highs = np.broadcast_to(constraint.ub, len(matrix))
            for row, low, high in zip(matrix, lows, highs):
                if low == high:
                    equal_rows.append(row)
                    equal_values.append(low)
                else:
                    if np.isfinite(high):
                        upper_rows.append(row)
                        upper_values.append(high)
                    if np.isfinite(low):
                        upper_rows.append(-row)
                        upper_values.append(-low)
        feasible = linprog(
            np.zeros(len(initial)),
            A_ub=np.asarray(upper_rows), b_ub=np.asarray(upper_values),
            A_eq=np.asarray(equal_rows) if equal_rows else None,
            b_eq=np.asarray(equal_values) if equal_values else None,
            bounds=[(None, None)] * len(initial), method="highs",
            options={"primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
        if not feasible.success:
            raise RuntimeError(f"Peak and shape constraints are infeasible: {feasible.message}")
        solution = minimize(
            objective, feasible.x, jac=objective_gradient,
            constraints=optimize_constraints, method="SLSQP",
            options={"maxiter": 1500, "ftol": 1e-12},
        )
    if not solution.success:
        raise RuntimeError(f"Quiet-phase Fourier fit failed: {solution.message}")
    c = lift(solution.x) * max_force
    coefficients = np.r_[complex(c[0]), c[1:1 + harmonic_count] - 1j * c[1 + harmonic_count:]]

    def continuous_bounds(start, end):
        grid_size = 32768
        y = evaluate_series(coefficients, np.linspace(start, end, grid_size + 1))
        # At an interior extremum f'=0, giving a second-derivative bound for
        # its distance from the nearest grid value. Endpoints are included.
        second_derivative_bound = (2 * np.pi)**2 * np.sum(k**2 * np.abs(coefficients[1:]))
        margin = second_derivative_bound / 8 * ((end - start) / grid_size)**2
        return float(y.min() - margin), float(y.max() + margin)

    bounds = continuous_bounds(0, 1)
    if peak_force_n is not None:
        # Locate stationary points of the finite Fourier polynomial. Unlike
        # the conservative grid bound, this can certify a peak on the exact
        # upper bound without reporting a spurious grid-margin overshoot.
        # Multiplying f'(phase) by z**H gives a polynomial in z=exp(i*2*pi*phase).
        derivative_polynomial = np.zeros(2 * harmonic_count + 1, dtype=complex)
        derivative_polynomial[harmonic_count + k] = k * coefficients[1:]
        derivative_polynomial[harmonic_count - k] = -k * coefficients[1:].conj()
        roots = np.polynomial.polynomial.polyroots(derivative_polynomial)
        unit_roots = roots[np.abs(np.abs(roots) - 1) < 1e-4]
        stationary_phase = np.mod(np.angle(unit_roots) / (2 * np.pi), 1)
        extrema_phase = np.r_[0, 1, peak_phase, stationary_phase]
        extrema = evaluate_series(coefficients, extrema_phase)
        dense = evaluate_series(coefficients, np.arange(32768) / 32768)
        bounds = (float(min(extrema.min(), dense.min())),
                  float(max(extrema.max(), dense.max())))
    quiet_bounds = continuous_bounds(0, quiet_end)
    tail_bounds = (None if tail_start_percent is None else
                   continuous_bounds(tail_start_percent / 100, 1))
    rise_derivative = basis(np.linspace(quiet_end, peak_phase, 20001), True) @ c
    fall_derivative_max = None
    if fall_end_percent is not None:
        fall_derivative_max = float(np.max(
            basis(np.linspace(peak_phase, fall_end_percent / 100, 20001), True) @ c
        ))
    force_tolerance = 1e-7 if peak_force_n is not None else 0
    fitted_peak = float(evaluate_series(coefficients, [peak_phase])[0])
    if (bounds[0] < -force_tolerance or bounds[1] > max_force + force_tolerance
            or quiet_bounds[1] > quiet_limit_n or rise_derivative.min() < -1e-5
            or (fall_derivative_max is not None and fall_derivative_max > 1e-3)
            or (tail_bounds is not None and tail_bounds[1] > tail_limit_n)
            or (peak_force_n is not None and abs(fitted_peak - peak_force_n) > 1e-7)):
        raise RuntimeError(
            "Continuous force bounds or monotonic-rise verification failed: "
            f"bounds={bounds}, quiet={quiet_bounds}, min_rise_derivative={rise_derivative.min()}, "
            f"max_fall_derivative={fall_derivative_max}, tail_bounds={tail_bounds}"
        )
    samples = evaluate_series(coefficients, sample_phase)
    return {
        "coefficients": coefficients,
        "frequency": np.arange(harmonic_count + 1) / period,
        "samples": samples,
        "scaled": scaled,
        "raw_gain": raw_gain,
        "peak_force_target_n": peak_force_n,
        "force_at_peak_phase_n": fitted_peak,
        "continuous_bounds_n": bounds,
        "quiet_bounds_n": quiet_bounds,
        "tail_bounds_n": tail_bounds,
        "minimum_rise_derivative_n_per_cycle": float(rise_derivative.min()),
        "maximum_fall_derivative_n_per_cycle": fall_derivative_max,
        "rms_error_n": float(np.sqrt(np.mean((samples - scaled)**2))),
    }


def make_visualizer(args):
    # Keep numerical postprocessing importable without simulation dependencies.
    import torch
    import yaml
    from env_wrapper_eval import GymMujocoWrapper
    from visualize import PolicyVisualizer
    from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
    from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11

    class AssistanceVisualizer(PolicyVisualizer):
        def load_config(self, cfg_path):
            self.device = torch.device("cpu")
            self.cfg = {
                "policy": {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "activation": "elu",
                    "terrain_cnn_enabled": False,
                    "exo_obs_noise_std": 0.0,
                },
                "visualization": {"save_name": "assistance_unused.txt"},
            }

        def setup_environment(self):
            with Path(args.env_config).open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            config.update(
                model_path=str(Path(args.xml).resolve()),
                render_mode=None,
                auto_render=False,
                terrain_obs_enabled=False,
                terrain_randomize_on_reset=False,
                log_reward_breakdown=False,
                enable_plthist=False,
                traj_seed=args.seed,
            )
            tm_args = config["traj_manager"]
            data_path = Path(args.data_path or tm_args["data_path"])
            if not data_path.is_absolute():
                data_path = ROOT / data_path
            manager = TrajectoryManager(
                data_path=data_path,
                sample_frequency=float(tm_args["sample_frequency"]),
                speed_range=tuple(tm_args["speed_range"]),
                verbose=False,
            )
            self.num_traj = manager.length()
            config["traj_manager"] = manager
            self.base_env = MFG_Musculoskeletal_V11(config, render_mode=None)
            self.env = GymMujocoWrapper(
                self.base_env, auto_reset=False, device=self.device
            )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return AssistanceVisualizer(None)


def collect_cycle(args):
    import mujoco
    import torch

    visualizer = make_visualizer(args)
    env = visualizer.base_env
    rows = []
    try:
        obs, info = visualizer.env.reset(id=args.traj_id, frame=0, seed=args.seed)
        if obs.shape[-1] != visualizer.num_actor_obs:
            raise ValueError(
                f"Environment observation {obs.shape[-1]} != checkpoint "
                f"{visualizer.num_actor_obs}; observation layouts must match."
            )
        ref = env.ref_traj
        if not ref.periodic:
            raise ValueError("The selected reference trajectory is not periodic.")
        cycle_frames = int(ref.step_frames)
        increment = int(ref.increment)
        if cycle_frames % increment:
            raise ValueError(
                f"Cycle frames {cycle_frames} are not divisible by increment {increment}; "
                "select a trajectory with a uniformly sampled complete cycle."
            )
        ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
               for name in ("ankle_motor_r", "ankle_motor_l")]
        if min(ids) < 0:
            raise ValueError("XML must contain cable actuators ankle_motor_r/l.")
        for idx in ids:
            if env.model.actuator_trntype[idx] != mujoco.mjtTrn.mjTRN_TENDON:
                raise ValueError("Expected tendon-transmission cable motors.")
            if env.model.actuator_gear[idx, 0] >= 0:
                raise ValueError("Expected negative cable gears (pulling tension).")
        sample_count = cycle_frames // increment
        dt = float(env.model.opt.timestep * env.frame_skip)
        target_start = args.cycle_index * cycle_frames
        target_end = target_start + cycle_frames
        if target_end >= ref.traj_frames:
            raise ValueError("Requested cycle extends beyond the reference clip.")
        target_forces = []
        for step in range((args.cycle_index + 1) * sample_count):
            frame = int(ref._pos)
            time_start = float(env.data.time)
            with torch.inference_mode():
                normal_obs = visualizer.obs_normalizer(
                    visualizer._compose_policy_obs(obs, info)
                )
                actions = visualizer._align_actions_to_env(
                    visualizer.policy.act_inference(normal_obs)
                )
                obs, reward, done, info = visualizer.env.step(actions)
            # Motor actuator_force is before transmission. Multiplication by
            # the negative cable gear gives tendon force; negate for tension.
            tension = -env.data.actuator_force[ids] * env.model.actuator_gear[ids, 0]
            rows.append([
                step, time_start, float(env.data.time), frame,
                frame // cycle_frames, (frame % cycle_frames) / cycle_frames * 100,
                *actions[0, ids].cpu().numpy().tolist(),
                *env.data.ctrl[ids].tolist(), *tension.tolist(),
            ])
            if target_start <= frame < target_end:
                target_forces.append(tension.copy())
            if bool(done.item()):
                # Do not silently combine cycles across reset/termination.
                raise RuntimeError(
                    f"Rollout terminated at frame {ref._pos} before export: "
                    f"{info.get('terminated_info', {})}"
                )
        if len(target_forces) != sample_count:
            raise RuntimeError("Could not record a complete gait cycle.")
        timing = np.asarray(rows, dtype=float)
        if not np.allclose(timing[:, 2] - timing[:, 1], dt, atol=1e-10, rtol=1e-8):
            raise RuntimeError("Simulation sampling interval is not uniform.")
        return np.asarray(target_forces), rows, {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "xml": str(Path(args.xml).resolve()),
            "seed": args.seed,
            "trajectory_id": args.traj_id,
            "speed_m_s": float(ref.speed),
            "cycle_index_zero_based": args.cycle_index,
            "reference_cycle_frames": cycle_frames,
            "reference_increment": increment,
            "reference_sample_frequency_hz": float(ref.sample_frequency),
            "samples_per_cycle": sample_count,
            "sampling_interval_s": dt,
            "sampling_rate_hz": 1 / dt,
            "cycle_duration_s": sample_count * dt,
            "phase_definition": "reference frame modulo cycle_frames / cycle_frames * 100",
            "sample_alignment": "force held over [time_start, time_end), labelled by action-input phase",
            "actuator_names": [env.model.actuator(i).name for i in ids],
            "actuator_gears": env.model.actuator_gear[ids, 0].tolist(),
            "raw_force_peak_n": np.max(target_forces, axis=0).tolist(),
            "rollout_terminated": False,
            "processing": "offline periodic reference; processed force was not applied in this rollout",
        }
    finally:
        visualizer.env.close()


def export_results(args, forces, rows, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = metadata["sampling_interval_s"]
    period = metadata["cycle_duration_s"]
    result = bandlimit_assistance(forces, dt, args.cutoff_hz, args.max_force)
    # Dense CSV includes phase 100% for convenient plotting; the FFT uses only
    # the N unique, uniformly spaced samples, without a duplicate endpoint.
    phase = np.linspace(0, 1, args.plot_points + 1)
    final = evaluate_series(result["coefficients"], phase)
    if final.min() < -1e-8 or final.max() > args.max_force + 1e-8:
        raise RuntimeError("Exported force bounds failed.")
    if max(result["high_frequency_ac_energy_fraction"]) > 1e-20:
        raise RuntimeError("Output has unexpected frequency energy above cutoff.")
    write_csv(out / "rollout.csv", [
        "step", "time_start_s", "time_end_s", "reference_frame", "cycle_index",
        "gait_phase_percent", "action_r", "action_l", "ctrl_r", "ctrl_l",
        "force_r_n", "force_l_n",
    ], rows)
    n = len(forces)
    write_csv(out / "cycle_samples.csv", [
        "time_s", "gait_phase_percent", "raw_r_n", "raw_l_n",
        "scaled_r_n", "scaled_l_n", "filtered_r_n", "filtered_l_n",
    ], np.column_stack((np.arange(n) * dt, np.arange(n) / n * 100,
                        forces, result["scaled"], result["samples"])))
    write_csv(out / "assistance_curve.csv", [
        "time_s", "gait_phase_percent", "force_r_n", "force_l_n",
    ], np.column_stack((phase * period, phase * 100, final)))
    freq = result["frequency"]
    raw_c = result["coefficients_raw"]
    final_c = result["coefficients"]
    write_csv(out / "fft.csv", [
        "frequency_hz", "raw_scaled_r_amplitude_n", "raw_scaled_l_amplitude_n",
        "filtered_r_amplitude_n", "filtered_l_amplitude_n",
    ], np.column_stack((freq, np.abs(raw_c), np.abs(final_c))))
    mask = result["retained"]
    # F(phi) = a0 + sum(a_k*cos(2*pi*k*phi) + b_k*sin(2*pi*k*phi)).
    write_csv(out / "fourier_coefficients.csv", [
        "harmonic", "frequency_hz", "cos_r_n", "sin_r_n", "cos_l_n", "sin_l_n",
    ], np.column_stack((np.arange(len(freq))[mask], freq[mask],
                        final_c[mask, 0].real, -final_c[mask, 0].imag,
                        final_c[mask, 1].real, -final_c[mask, 1].imag)))
    metadata.update({
        "cutoff_hz": args.cutoff_hz,
        "force_limit_n": args.max_force,
        "retained_max_frequency_hz": float(freq[mask][-1]),
        "retained_max_harmonic": int(np.flatnonzero(mask)[-1]),
        "frequency_resolution_hz": 1 / period,
        "raw_force_to_scaled_gain": result["raw_gain"],
        "post_filter_dc_offset_n": result["dc_offset_n"],
        "post_filter_shared_gain": result["final_gain"],
        "continuous_lower_bound_n": result["continuous_lower_bound_n"],
        "continuous_upper_bound_n": result["continuous_upper_bound_n"],
        "high_frequency_ac_energy_fraction": result["high_frequency_ac_energy_fraction"],
        "filter": "periodic FFT hard cutoff; DC offset and common gain enforce bounds without clipping",
        "fourier_formula": "F(phi)=cos_0+sum(cos_k*cos(2*pi*k*phi)+sin_k*sin(2*pi*k*phi)); phi in cycles",
        "fft_convention": "single-sided peak amplitudes; DC undoubled; no window on one periodic cycle",
    })
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    colors = ["#0072B2", "#D55E00"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.plot(np.arange(n) / n * 100, result["scaled"][:, j], "o--",
                color="0.65", markersize=3, linewidth=1,
                label=f"Model force scaled within {args.max_force:g} N")
        ax.plot(phase * 100, final[:, j], color=colors[j], linewidth=2,
                label=f"FFT low-pass <= {args.cutoff_hz:g} Hz, bounded 0-{args.max_force:g} N")
        ax.set(ylabel=f"{'Right' if j == 0 else 'Left'} cable force (N)",
               ylim=(-3, args.max_force * 1.08), xlim=(0, 100))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Reference gait phase (%)")
    fig.suptitle(
        f"Flat cable assistance | cycle {args.cycle_index + 1} | "
        f"T = {period:.3f} s | speed = {metadata['speed_m_s']:.2f} m/s"
    )
    fig.savefig(out / "force_gait_phase.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.plot(freq, np.abs(raw_c[:, j]), "o--", color="0.6", markersize=3,
                label="Scaled model force")
        ax.plot(freq, np.abs(final_c[:, j]), "o-", color=colors[j], markersize=3,
                label="Exported assistance")
        ax.axvline(args.cutoff_hz, color="#009E73", linestyle=":",
                   label=f"{args.cutoff_hz:g} Hz cutoff")
        ax.set(ylabel=f"{'Right' if j == 0 else 'Left'} amplitude (N)",
               xlim=(0, 0.5 / dt))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"One-cycle FFT | df = {1 / period:.3f} Hz | DC included")
    fig.savefig(out / "fft.png", dpi=200)
    plt.close(fig)
    (out / "README.md").write_text(
        "# Flat cable assistance\n\n"
        "Generated by `export_assistance_curve.py` from the checkpoint and XML "
        "listed in `metadata.json`.\n\n"
        "- `force_gait_phase.png`: right/left scaled model force and processed reference.\n"
        "- `fft.png`, `fft.csv`: single-sided FFT amplitudes, including DC.\n"
        "- `assistance_curve.csv`: dense periodic reference (N), with a duplicate "
        "100% endpoint for plotting.\n"
        "- `cycle_samples.csv`: original cycle samples, scaled and filtered forces.\n"
        "- `rollout.csv`: controls and measured transmitted forces for all recorded cycles.\n"
        "- `fourier_coefficients.csv`: coefficients for continuous evaluation.\n\n"
        "## Reproduce\n\n"
        "Run from the project directory in an environment with the project dependencies:\n\n"
        "```powershell\npython export_assistance_curve.py\n```\n\n"
        "With the workspace's `.test-deps` and current Python:\n\n"
        "```powershell\n"
        "python -c \"import sys,runpy; sys.path.insert(0,r'.test-deps'); "
        "runpy.run_path('export_assistance_curve.py',run_name='__main__')\"\n"
        "```\n\n"
        "## Interpretation\n\n"
        f"Cycle {args.cycle_index + 1} is used (zero-based index {args.cycle_index}). "
        f"Period = {period:.6g} s; original sampling = {1 / dt:.6g} Hz. "
        "Phase follows the reference trajectory, rather than measured heel strikes. "
        "The force is the tensile load in each cable, not ankle torque.\n\n"
        f"The FFT retains only frequencies <= {args.cutoff_hz:g} Hz. "
        "A per-leg DC offset removes any negative ringing; a common gain then "
        f"bounds both continuous curves within 0..{args.max_force:g} N. "
        "No pointwise clipping is applied after filtering. This preserves the "
        "bandwidth but can add a small baseline force during low-assistance phases.\n\n"
        "For phase `phi` in [0,1), evaluate each leg as:\n\n"
        "`F(phi) = cos_0 + sum(cos_k*cos(2*pi*k*phi) + sin_k*sin(2*pi*k*phi))`.\n\n"
        "The bandwidth applies to continuous periodic playback at the recorded "
        "period; changing phase speed changes the temporal frequencies. "
        "FFT bins use the original unique samples, without the duplicate endpoint. "
        "The dense CSV is Fourier interpolation and does not add simulation samples.\n\n"
        "These are offline processed references. The rollout used the original "
        "policy controls; it does not validate walking with the processed reference.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"Results saved to {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "models/exo_flat_terrain/model_best.pt"))
    parser.add_argument("--xml", default=str(ROOT / "Env/MFG_Musculoskelet_V11/MFG_Raj2015_V11_cable.xml"))
    parser.add_argument("--env-config", default=str(ROOT / "Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml"))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--traj-id", type=int, default=0)
    parser.add_argument("--cycle-index", type=int, default=1, help="Zero-based; 1 skips the initial cycle.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    parser.add_argument("--max-force", type=float, default=100.0)
    parser.add_argument("--plot-points", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(ROOT / "plots_exo_terrain/flat_assistance_100N_8Hz"))
    args = parser.parse_args()
    if args.cycle_index < 0 or args.plot_points < 4:
        parser.error("cycle-index must be nonnegative and plot-points >= 4")
    forces, rows, metadata = collect_cycle(args)
    export_results(args, forces, rows, metadata)


if __name__ == "__main__":
    main()
