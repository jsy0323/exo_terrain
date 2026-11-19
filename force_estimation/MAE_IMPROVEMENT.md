# Hip-moment MAE fine-tuning experiment

The selected estimator is `models/force_estimator_mae_ablation/smooth_l1_low_aux/model_best.pt` (epoch 63). Its exact configuration is in the same directory as `config.yaml`. Outputs, sensor preprocessing, sample rate, geometry, data split and 30 Hz causal label bandwidth match the original simulation experiment.

All three variants start from the same epoch-43 checkpoint, reset Adam, restore the same saved RNG states and use 25 additional epochs at learning rate `1e-4`. Initialization participates in checkpoint selection. Each variant is selected by mean left/right hip MAE on validation mass profile 0.97. The experiment commits `selection.json` before opening the test recordings. The held-out mass profile 1.09 is then evaluated for all variants, including the initial checkpoint reference.

| Variant | Selected epoch | Validation hip MAE L/R (Nm) | Test hip MAE L/R (Nm) |
| --- | --- | --- | --- |
| Initial checkpoint | 43 | 3.24 / 2.73 | 3.89 / 2.67 |
| Huber control fine-tuning | 61 | 2.91 / 2.41 | 3.04 / 2.32 |
| SmoothL1, original auxiliary weights | 63 | 2.90 / 2.39 | 3.10 / 2.23 |
| **SmoothL1, lower auxiliary weights** | **63** | **2.83 / 2.38** | **3.04 / 2.25** |

The selected variant uses normalized hip SmoothL1 with `beta=2/50`, corresponding to a 2 Nm transition. Weights are hip 1.0, knee/ankle 0.1, GRF 0.05, contact 0.05, phase 0.05, and Gaussian NLL 0.02. Uncertainty scales are calibrated on validation residuals after selecting the checkpoint.

Compared with initialization, selected-model test MAE decreases by 21.7%/15.7%, or 19.3% for the mean of both hips. Test RMSE is 4.20/3.29 Nm and absolute-error P95 is 8.49/7.25 Nm. Against instantaneous raw biological moment, MAE is 5.12/5.62 Nm. Most of the MAE gain is also obtained by the low-learning-rate Huber control; the three fine-tuned variants are close, and this is one paired optimization seed. These results do not isolate a large architecture or loss-function advantage. This test partition was also reported during earlier development; it provides a consistent comparison rather than a newly collected independent audit. Additional seeds and fresh held-out recordings are needed to measure robustness of the gain.

Calibrated +/-2 sigma test coverage is 93.0%/96.0%. Left bias remains approximately -1.16 Nm. Auxiliary test MAEs are knees 3.29/1.52 Nm, ankles 7.11/6.11 Nm, and GRF 76.60/62.64 N; contact accuracy is 97.60%. The model retains the original network structure and causal window. CPU single-thread median/P95 runtime is 3.69/3.87 ms over 201 valid streamed frames; streamed and batched hip predictions differ by at most 0.000031 Nm.

Six paired ideal-torque closed-loop trials (two seeds each at gains 0, 0.1 and 0.2, 12 seconds) complete without a rejected episode. At gain 0.2, hip estimation MAE is 3.51/3.50 Nm and requested torques exceed the 4 Nm limit for 60.1%/44.4% of ready samples. Results are in `logs/force_estimation_mae_closed_loop/summary.json`. Interpretation remains limited to the simulated walking policy and ideal torque transmission.

## Run or inspect

```powershell
conda activate rsl
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

# Re-entering the completed experiment verifies its plan and reuses completed training.
python -m force_estimation.ablate_mae

# Export the selected estimator's five figures, SVGs, and multipage PDF.
python -m force_estimation.plot_results --checkpoint models/force_estimator_mae_ablation/smooth_l1_low_aux/model_best.pt

# Repeat closed-loop experiments in a fresh directory.
python -m force_estimation.closed_loop --checkpoint models/force_estimator_mae_ablation/smooth_l1_low_aux/model_best.pt --output-dir logs/force_estimation_mae_closed_loop_repeat
```

The original `closed_loop` and plotting CLI defaults still point to the original simulation estimator; pass the checkpoint explicitly as above. The new estimator loads through the existing `OnlineMomentEstimator` interface with the subject measurements and per-device FSR calibration curves.

Experiment artifacts are `plan.json`, `selection.json`, `summary.json` and one training directory per variant under `models/force_estimator_mae_ablation`. Selected-model figures and prediction arrays are in `smooth_l1_low_aux/results`.

The estimator tests pass in the rsl environment (MuJoCo 3.11). The full suite passes 51 tests using the project's `.test-deps` environment (MuJoCo 3.3.2). In rsl's MuJoCo 3.11, the full suite has 50 passing tests and one failing external-exoskeleton cuff-travel test (`SoftContactTest.test_supported_gait_poses_stay_inside_cuff_travel`); that test does not exercise the estimator.
