# exo_terrain

[![CI](https://github.com/jsy0323/exo_terrain/actions/workflows/ci.yml/badge.svg)](https://github.com/jsy0323/exo_terrain/actions/workflows/ci.yml)

Reinforcement learning of human–exoskeleton interaction in MuJoCo: a muscle-driven musculoskeletal human model wearing a cable-driven lower-limb exoskeleton, trained on walking and running over flat, sloped, stair and random terrain.

## Repository layout

| Path | Contents |
| --- | --- |
| `Env/MFG_Musculoskelet_V11/` | MuJoCo models (human, exoskeleton variants, terrain), environment code (`mfgenv/`), reference motion data and terrain generators (`tools/`) |
| `RL/` | PPO and distillation algorithms, actor–critic networks, rollout storage and the on-policy runner |
| `force_estimation/` | Simulated wearable sensors and a causal estimator of hip, knee and ankle joint moments — see [force_estimation/README.md](force_estimation/README.md) |
| `configs/` | Training, visualization and force-estimation configurations |
| `tests/` | Unit and integration tests, run automatically by CI |

## Usage

```bash
pip install -r requirements.txt

# Train (Hydra config: configs/default_V11.yaml)
python trainV11.py

# Visualize a trained policy
python visualize.py --config configs/visualize.yaml
```

## Tests

```bash
python -m pytest -q
```

Every push and pull request runs the test suite on Ubuntu through [GitHub Actions](.github/workflows/ci.yml), using the pinned dependencies in `requirements-ci.txt`. Tests that need a locally trained policy checkpoint (`logs/humanoid_walk_run/model_best.pt`, not tracked in git) are skipped when it is absent.

## License

[MIT](LICENSE)
