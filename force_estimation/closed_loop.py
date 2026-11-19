"""Paired ideal-joint-source experiments, before a mechanical exoskeleton exists."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .collect import collect


def run(config, checkpoint, output_dir, gains=(0., .1, .2), episodes=2, duration_s=12.):
    config = copy.deepcopy(config)
    config["experiments"]["subjects"] = [next(profile for profile in config["experiments"]["subjects"]
                                                   if profile["id"] == "mass_100")]
    config["experiments"]["motions"] = [next(profile for profile in config["experiments"]["motions"]
                                                  if profile["id"] == "steady")]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for gain in gains:
        directory = output / f"gain_{gain:.3f}"
        manifest = collect(config, episodes, directory, duration_s, estimator_checkpoint=checkpoint, assistance_gain=gain)
        errors, biological, applied, speeds, saturated = [], [], [], [], []
        failures = 0
        for entry in manifest["episodes"]:
            with np.load(directory / entry["file"], allow_pickle=False) as data:
                ready = data["estimator_ready"].astype(bool)
                if not ready.any():
                    failures += 1
                    continue
                errors.append(data["estimated_tau_bio_hip"][ready] - data["tau_bio"][ready, :2])
                biological.append(data["tau_bio"][ready, :2])
                applied.append(data["tau_applied_raw"][ready, :2])
                speeds.append(data["forward_speed"][ready])
                saturated.append(np.abs(data["requested_hip_torque"][ready]) > config["experiments"]["torque_limit_nm"])
                failures += int(not entry["accepted"])
        if not errors:
            results.append({"gain": gain, "failed_episodes": failures, "samples": 0})
            continue
        error, biology, torque = np.concatenate(errors), np.concatenate(biological), np.concatenate(applied)
        result = {"gain": gain, "samples": len(error), "failed_episodes": failures,
                  "hip_estimation_mae_nm": np.mean(np.abs(error), axis=0).tolist(),
                  "biological_torque_rms_nm": np.sqrt(np.mean(biology ** 2, axis=0)).tolist(),
                  "applied_torque_max_abs_nm": np.max(np.abs(torque), axis=0).tolist(),
                  "requested_torque_saturation_fraction": np.concatenate(saturated).mean(axis=0).tolist(),
                  "speed_mean_mps": float(np.mean(np.concatenate(speeds)))}
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/force_estimation_sim.yaml")
    parser.add_argument("--checkpoint", default="models/force_estimator_sim_v2/model_best.pt")
    parser.add_argument("--output-dir", default="logs/force_estimation_closed_loop")
    parser.add_argument("--gains", nargs="+", type=float, default=[0., .1, .2])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=12.)
    arguments = parser.parse_args()
    with open(arguments.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run(config, arguments.checkpoint, arguments.output_dir, arguments.gains, arguments.episodes, arguments.duration_s)


if __name__ == "__main__":
    main()
