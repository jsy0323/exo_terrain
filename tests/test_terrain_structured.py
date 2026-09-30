import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Env.MFG_Musculoskelet_V11.mfgenv.terrain import (
    build_structured_heightmap_m,
    get_local_height_patch_m,
)


class StructuredTerrainTest(unittest.TestCase):
    def test_structured_heightmap_has_flat_spawn_and_mixed_segments(self):
        height = build_structured_heightmap_m(
            nrow=101,
            ncol=1001,
            x_half=50.0,
            height_scale=0.3,
            rng=np.random.default_rng(7),
            flat_start_length=0.5,
        )

        profile = height[height.shape[0] // 2]
        x = np.linspace(-50.0, 50.0, profile.size)
        delta = np.diff(profile)

        self.assertEqual(height.shape, (101, 1001))
        self.assertTrue(np.allclose(profile[x <= 0.5], 0.0))
        self.assertTrue(np.allclose(height[0], height[-1]))
        self.assertGreaterEqual(float(height.min()), 0.0)
        self.assertLessEqual(float(height.max()), 0.3 + 1e-6)
        self.assertGreater(np.count_nonzero(np.isclose(delta, 0.0)), 0)
        self.assertGreater(np.count_nonzero(~np.isclose(delta, 0.0)), 0)

    def test_structured_heightmap_is_repeatable_with_seeded_rng(self):
        kwargs = {
            "nrow": 11,
            "ncol": 101,
            "x_half": 5.0,
            "height_scale": 0.3,
            "flat_start_length": 0.5,
        }

        first = build_structured_heightmap_m(rng=np.random.default_rng(11), **kwargs)
        second = build_structured_heightmap_m(rng=np.random.default_rng(11), **kwargs)

        self.assertTrue(np.array_equal(first, second))

    def test_structured_heightmap_can_force_flat_only(self):
        height = build_structured_heightmap_m(
            nrow=5,
            ncol=101,
            x_half=5.0,
            height_scale=0.3,
            rng=np.random.default_rng(3),
            flat_start_length=0.0,
            structured_kinds="flat",
        )

        self.assertTrue(np.allclose(height, 0.0))

    def test_structured_heightmap_can_force_stairs_only(self):
        height = build_structured_heightmap_m(
            nrow=5,
            ncol=201,
            x_half=5.0,
            height_scale=0.5,
            rng=np.random.default_rng(3),
            flat_start_length=0.0,
            segment_length_range=(5.0, 5.0),
            step_height_range=(0.05, 0.05),
            step_run_range=(0.5, 0.5),
            structured_kinds=["stairs"],
        )

        profile = height[height.shape[0] // 2]
        delta = np.diff(profile)
        self.assertGreater(np.count_nonzero(~np.isclose(delta, 0.0)), 0)
        self.assertGreater(np.count_nonzero(np.isclose(delta, 0.0)), 0)

    def test_structured_heightmap_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            build_structured_heightmap_m(
                nrow=5,
                ncol=101,
                x_half=5.0,
                height_scale=0.3,
                structured_kinds="unknown",
            )

    def test_local_patch_is_flattened_as_lateral_rows_by_forward_columns(self):
        env = SimpleNamespace(
            model=SimpleNamespace(body=lambda _: SimpleNamespace(id=0)),
            data=SimpleNamespace(xpos=np.array([[0.0, 0.0, 0.0]])),
            pelvis_heading=np.array([1.0, 0.0, 0.0]),
        )

        with patch(
            "Env.MFG_Musculoskelet_V11.mfgenv.terrain.sample_hfield_height_m",
            side_effect=lambda _model, _data, x, y, **_kwargs: x + 10.0 * y,
        ):
            patch_values = get_local_height_patch_m(
                env,
                grid_x=(0.0, 1.0, 2.0),
                grid_y=(-1.0, 1.0),
                relative_to_center=False,
            )

        self.assertTrue(
            np.array_equal(
                patch_values.reshape(2, 3),
                np.array([[-10.0, -9.0, -8.0], [10.0, 11.0, 12.0]], dtype=np.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()
