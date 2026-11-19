This package collects simulated wearable data and trains a causal estimator of bilateral net hip, knee and ankle joint moments. The default control interpretation is assistance relative to the person's own contribution.

The expanded simulation workflow with AHRS, anti-alias filtering, virtual FSR calibration, held-out mass profiles and ideal-source feedback is described in [SIMULATION.md](SIMULATION.md). Its ready-to-use checkpoint is `models/force_estimator_sim_v2/model_best.pt` and its configuration is `configs/force_estimation_sim.yaml`.

The current training configuration uses human-only sensing with `training.include_exo_features=false`. Its 54 additional features comprise encoder features (4), FSR features (14), kinematics (18), physics proxies (6), subject parameters (5), and IMU availability flags (7). The 64-dimensional pooled IMU token and these features form a 118-dimensional TCN input before projection. An unassisted person has `tau_bio=tau_total`, so this acquisition stage can proceed before designing the exoskeleton.

For consistent signs and the same anatomical joint axis,

\[
\tau_{bio}=\tau_{total}-\tau_{exo\to human},\qquad
\tau_{exo,cmd}=a\,\hat\tau_{bio}.
\]

An ideal static balance gives an exoskeleton share of total demand of `a/(1+a)`. An `a` of 0.3 therefore corresponds to about 23% of total demand. Delivered torque needs a measured or calibrated estimate at the human joint. The estimator and actuator command must use the same time alignment and axis convention. This package supplies estimation software and datasets.

Activate the existing `rsl` environment in the repository root. Collect the configured 30 episodes and train with the following commands. Each command accepts `--help`.

```powershell
conda activate rsl
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
python -m force_estimation.collect --config configs/force_estimation.yaml
python -m force_estimation.train --config configs/force_estimation.yaml
```

`--episodes`, `--duration-s` and `--output-dir` override acquisition settings. Training accepts `--data-dir`, `--output-dir` and `--epochs`. An existing dataset or model directory containing completed output is protected against replacement.

Continue the existing human-only estimator for 30 additional epochs with the following command. The original model has completed 15 epochs, so the new run reaches epoch 45. The checkpoint directory supplies the default output directory and the saved episode partition identifies the dataset. Normalization statistics and the train/validation/test assignment are restored.

```powershell
python -m force_estimation.train --resume models/force_estimator_human_only/model_best.pt --epochs 30 --learning-rate 1e-4
```

The command uses a learning rate of `1e-4` for fine-tuning. Each completed epoch saves `model_last.pt`, optimizer state, random-number-generator states and training history. `model_best.pt` is selected using validation loss. To continue from the latest completed epoch, use `--resume models/force_estimator_human_only/model_last.pt`. Earlier prototype checkpoints contain model parameters and normalization statistics. Their first continuation initializes a fresh Adam optimizer because the earlier files have no optimizer state. Passing a new `--output-dir` branches the continuation into that directory.

The starter run can be reproduced in new output directories with these commands.

```powershell
python -m force_estimation.collect --episodes 8 --output-dir data/force_estimation_starter_v2
python -m force_estimation.train --data-dir data/force_estimation_starter_v2 --output-dir models/force_estimator_starter_v2 --epochs 15
```

Collection reuses `visualize.PolicyVisualizer` and the policy checkpoint in `configs/force_estimation.yaml`. Seven massless sites, gyro sensors and accelerometers are added in memory before compiling the model. The policy retains its original observation and action layout. Wearable sensing uses synchronized 1 ms physics states and interpolated per-IMU delays, then produces uniform 100 Hz packets. Calibration frames have +X forward, +Y left, +Z up at the model's neutral pose. Mounting position and Euler-angle offsets are resampled each episode. Noise, bias and vibration amplitudes are per sensor axis.

The current V11 feet have separate rearfoot and forefoot collision geometries. Their ground contact points are partitioned in calibrated foot coordinates with randomized boundaries. FSR inputs are summed contact-normal forces. Vertical GRF labels use the signed world-vertical component, including the effect of friction. The provided FSR response is divided by `calibration_value=500`, then bias and noise are added in normalized units. The 0.3–1.0 lag coefficient refers to a 20 ms control step and is converted to the 10 ms sample interval using `1-(1-k)^(dt/0.02)`.

IMU packets contain gyro, specific force and unit gravity direction in sensor coordinates. The simulated gravity direction uses the true site orientation, as specified in the acquisition design. Physical deployment must provide its causal AHRS gravity estimate in the same calibrated frame. AHRS error is an additional sim-to-real difference to characterize with physical recordings. [MuJoCo sensor conventions](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer)

Every episode NPZ contains the following arrays. Side order is left then right throughout. Joint order is left/right hip flexion, left/right knee angle, left/right ankle angle.

| Array | Shape | Meaning |
| --- | --- | --- |
| `time` | T | Seconds from reset, float64 |
| `imu`, `imu_ideal` | T × 7 × 9 | Randomized/delayed packet and clean current sensor values |
| `imu_mask` | T × 7 | Sensor availability |
| `encoder` | T × 4 | Two measured angles, then two causal filtered rates |
| `fsr`, `fsr_force_n` | T × 4 | Heel/toe left, heel/toe right |
| `extra` | T × 54 | Human-only features listed in `features.HUMAN_EXTRA_NAMES` |
| `tau_total`, `tau_bio`, `tau_exo` | T × 6 | Net joint moments in N m |
| `grf_v`, `contact` | T × 2 | Vertical force in N and thresholded contact |
| `phase`, `phase_mask` | T × 2 × 2, T × 2 | Per-leg sine/cosine phase and valid-stride mask |
| `metadata` | scalar JSON string | Seed, subject, mounting, delay and FSR parameters |

Net joint demand is computed at the synchronized physics state using `M(q) @ qacc + qfrc_bias - qfrc_constraint`. This includes active and passive non-constraint contributions. The dataset records the exact formula and MJCF axis signs in its manifest. MuJoCo's `qfrc_inverse` additionally removes passive forces, so the recorded quantity is computed explicitly. The force-balance regression checks it against active, passive and applied generalized forces. [MuJoCo dynamics conventions](https://mujoco.readthedocs.io/en/stable/computation/index.html)

The current collector uses an unassisted humanoid. `tau_exo` is stored as zero and `tau_bio` equals `tau_total`. Powered acquisition will need a delivered-torque measurement adapter and powered examples before the estimator can be evaluated in that condition. Body mass comes from the model. Height and segment lengths are editable measured-subject inputs, currently nominal values in the configuration. The starter data describe one simulated subject and the motions actually performed by the selected locomotion checkpoint. Training selects columns by the manifest's feature names and can reuse the earlier 62-feature starter data. The online loader supports both 54-feature human-only and earlier 62-feature checkpoints.

Stride phase labels use actual rising contact edges. Samples outside complete consecutive-strike intervals are masked from phase loss. Future strikes are used only for offline labels. All network inputs and feature calculations are causal.

Four IMU encoders serve the pelvis and three bilateral segment pairs. Right-side gyro vectors are mirrored as axial vectors using `[-1,+1,-1]`, while acceleration and gravity are mirrored as polar vectors using `[+1,-1,+1]`. Learnable part embeddings distinguish the seven tokens. Masked attention returns zero fused input when all IMUs are unavailable. The fused 64-vector is concatenated with the configured additional features. Six 128-channel causal convolution layers with kernel 3 and dilations 1–32 give a receptive field of 127 samples, or 1.27 s at 100 Hz. The default input window is 100 samples. Temporal convolutions use left padding, and normalization acts within each time instant.

Loss weights follow the requested `1.0, 0.3, 0.2, 0.2, 0.2, 0.1` ordering. Torque and GRF regression losses are computed in configured scaled units, with defaults 50 N m and 750 N. Public outputs retain physical units. The heteroscedastic Gaussian NLL trains a positive hip error standard deviation. `sigma_nm` is in N m and `confidence=1/(1+sigma_nm)` follows the requested numerical convention. Confidence and sigma coverage must be evaluated for the deployment population and operating conditions.

Training partitions entire episodes before making windows and fits feature statistics using training episodes only. A validation partition selects `model_best.pt`, followed by one test evaluation. The model directory includes its schema, normalization, target definition, episode partition, training history and held-out error/uncertainty metrics. Episode separation evaluates held-out acquisition runs of this one subject.

`OnlineMomentEstimator` shares the encoder filter and feature builder with acquisition. Supply uniformly resampled, synchronized packets and measured subject values. It returns `None` during the first 99 samples, then provides estimates at each new sample. Call `reset()` after discontinuities and at the start of a recording.

```python
from force_estimation.runtime import OnlineMomentEstimator

estimator = OnlineMomentEstimator(
    "models/force_estimator_human_only/model_best.pt",
    subject=[mass_kg, height_m, thigh_length_m, shank_length_m, foot_length_m],
)
prediction = estimator.step(timestamp, imu_7x9, imu_mask_7, hip_angles_2, fsr_4)
# prediction["tau_hip"] is [left, right] in N m for the saved target definition.
# prediction["sigma_nm"] and prediction["confidence"] have the same side order.
```
