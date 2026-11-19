import unittest

import numpy as np

from export_assistance_curve import bandlimit_assistance, evaluate_series, fit_assistance_quiet_phase


class AssistanceCurveTests(unittest.TestCase):
    def test_removes_high_frequency_preserving_low_harmonic(self):
        time = np.arange(100) / 100
        force = 40 + 15 * np.cos(2 * np.pi * 3 * time) + 8 * np.sin(2 * np.pi * 19 * time)
        result = bandlimit_assistance(np.column_stack((force, force / 2)), 0.01)
        c = result["coefficients"]
        self.assertAlmostEqual(abs(c[3, 0]), 15 * result["final_gain"], places=10)
        self.assertTrue(np.all(c[result["frequency"] > 8] == 0))
        # Both legs receive the same multiplicative scale.
        np.testing.assert_allclose(c[3, 0], 2 * c[3, 1], atol=1e-12)

    def test_continuous_bounds_and_periodic_endpoints_for_odd_and_even_lengths(self):
        for n in (53, 64):
            with self.subTest(n=n):
                forces = np.zeros((n, 2))
                forces[2:5, 0] = 200
                forces[11:20, 1] = 160
                result = bandlimit_assistance(forces, 0.02)
                dense = evaluate_series(result["coefficients"], np.linspace(0, 1, 100003))
                self.assertGreaterEqual(dense.min(), -1e-10)
                self.assertLessEqual(dense.max(), 100 + 1e-10)
                np.testing.assert_allclose(dense[0], dense[-1], atol=1e-10)
                self.assertGreater(max(result["dc_offset_n"]), 0)
                self.assertLess(max(result["high_frequency_ac_energy_fraction"]), 1e-25)

    def test_constant_force_has_only_dc_and_is_not_scaled_up(self):
        result = bandlimit_assistance(np.full((50, 2), 25.0), 0.02)
        np.testing.assert_allclose(result["samples"], 25, atol=1e-10)
        np.testing.assert_allclose(result["coefficients"][1:], 0, atol=1e-10)

    def test_invalid_force_or_sampling_is_rejected(self):
        forces = np.ones((50, 2))
        for dt, cutoff, limit in ((0.0, 8, 100), (0.1, 8, 100), (0.02, 8, -1),
                                  (0.02, 8, float("nan"))):
            with self.subTest(dt=dt, cutoff=cutoff, limit=limit):
                with self.assertRaises(ValueError):
                    bandlimit_assistance(forces, dt, cutoff, limit)
        for invalid in (np.zeros((50, 2)), -forces, np.full((50, 2), np.nan)):
            with self.assertRaises(ValueError):
                bandlimit_assistance(invalid, 0.02)

    def test_quiet_phase_fit_suppresses_pre_rise_without_high_frequencies(self):
        # A pulse with a silent initial quarter cycle and a noisy falling edge.
        n = 49
        phase = np.arange(n) / n
        force = np.where((phase > .27) & (phase < .65),
                         180 * np.sin(np.pi * np.clip((phase - .27) / .38, 0, 1)), 0)
        force[-2:] = [40, 90]  # Deliberately mismatched periodic boundary.
        result = fit_assistance_quiet_phase(force, .02, 25, 44, quiet_limit_n=.5,
                                           fall_end_percent=65)
        dense_phase = np.arange(65536) / 65536
        dense = evaluate_series(result["coefficients"], dense_phase)
        self.assertGreaterEqual(dense.min(), 0)
        self.assertGreater(dense.max(), 75)
        self.assertLessEqual(dense.max(), 100)
        self.assertLessEqual(dense[dense_phase <= .25].max(), .5)
        rise = dense[(dense_phase >= .25) & (dense_phase <= .44)]
        self.assertGreaterEqual(np.diff(rise).min(), -1e-7)
        fall = dense[(dense_phase >= .44) & (dense_phase <= .65)]
        self.assertLessEqual(np.diff(fall).max(), 1e-7)
        self.assertAlmostEqual(dense_phase[dense.argmax()], .44, delta=1e-3)
        spectrum = np.fft.rfft(dense)
        frequency = np.fft.rfftfreq(len(dense), (n * .02) / len(dense))
        self.assertLess(np.abs(spectrum[frequency > 8]).max(), 1e-7)
        endpoints = evaluate_series(result["coefficients"], [0, 1])
        self.assertAlmostEqual(endpoints[0], endpoints[1], places=10)

    def test_exact_peak_constraint_preserves_quiet_interval_and_force_limit(self):
        phase = np.arange(49) / 49
        force = np.where((phase > .27) & (phase < .65),
                         180 * np.sin(np.pi * np.clip((phase - .27) / .38, 0, 1)), 0)
        force[-2:] = [40, 90]
        result = fit_assistance_quiet_phase(
            force, .02, 25, 44, quiet_limit_n=.5,
            fall_end_percent=65, peak_force_n=100,
        )
        self.assertAlmostEqual(evaluate_series(result["coefficients"], [.44])[0], 100, places=9)
        dense_phase = np.arange(100003) / 100003
        dense = evaluate_series(result["coefficients"], dense_phase)
        self.assertLessEqual(dense.max(), 100 + 1e-7)
        self.assertGreaterEqual(dense.min(), 0)
        self.assertLessEqual(dense[dense_phase <= .25].max(), .5)
        self.assertLessEqual(result["frequency"][-1], 8)

    def test_invalid_peak_amplitudes_are_rejected(self):
        for peak in (0, .5, 101, float("nan")):
            with self.subTest(peak=peak):
                with self.assertRaises(ValueError):
                    fit_assistance_quiet_phase(np.ones(49), .02, 25, 44, peak_force_n=peak)

    def test_tail_peak_is_suppressed_without_moving_the_main_peak(self):
        phase = np.arange(49) / 49
        force = np.where((phase > .27) & (phase < .67),
                         180 * np.exp(-((phase - .41) / .10)**2), 0)
        force[(phase > .8) & (phase < .93)] = 70
        result = fit_assistance_quiet_phase(
            force, .02, 26.53061224489796, 40.1, quiet_limit_n=.9,
            fall_end_percent=65.3061224489796, peak_force_n=100,
            tail_start_percent=75, tail_limit_n=.5,
        )
        grid = np.linspace(0, 1, 100001)
        fitted = evaluate_series(result["coefficients"], grid)
        self.assertAlmostEqual(evaluate_series(result["coefficients"], [.401])[0], 100, places=9)
        self.assertLessEqual(fitted[grid >= .75].max(), .5)
        self.assertLessEqual(fitted[grid <= .2653061224489796].max(), .9)
        self.assertGreaterEqual(fitted.min(), 0)
        self.assertLessEqual(fitted.max(), 100 + 1e-7)
        fall = fitted[(grid >= .401) & (grid <= .653061224489796)]
        self.assertLessEqual(np.diff(fall).max(), 1e-7)
        self.assertLessEqual(result["frequency"][-1], 8)


if __name__ == "__main__":
    unittest.main()
