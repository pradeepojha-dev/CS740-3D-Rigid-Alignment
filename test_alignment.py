"""Run: python -m unittest -v test_alignment.py"""
import unittest
import numpy as np
from alignment import (estimate_svd, estimate_quaternion, transform,
                       axis_angle_to_rotation, rotation_error_degrees,
                       alignment_rmse, geometry_ratio)


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)
        self.X = self.rng.normal(size=(40, 3))
        self.R = axis_angle_to_rotation([0.3, -0.5, 0.8], 42)
        self.t = np.array([0.4, -0.6, 0.25])
        self.Y = transform(self.X, self.R, self.t)

    def _assert_exact_fit(self, fn, X=None, Y=None, R=None, t=None):
        x = self.X if X is None else X
        y = self.Y if Y is None else Y
        true_r = self.R if R is None else R
        true_t = self.t if t is None else t
        r, trans = fn(x, y)
        np.testing.assert_allclose(r, true_r, atol=1e-10)
        np.testing.assert_allclose(trans, true_t, atol=1e-10)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(r), 1, places=10)
        self.assertLess(alignment_rmse(x, y, r, trans), 1e-10)

    def test_svd_noiseless(self):
        self._assert_exact_fit(estimate_svd)

    def test_quaternion_noiseless(self):
        self._assert_exact_fit(estimate_quaternion)

    def test_identity(self):
        for f in (estimate_svd, estimate_quaternion):
            self._assert_exact_fit(f, self.X, self.X, np.eye(3), np.zeros(3))

    def test_nonzero_centroids(self):
        x = self.X + [20., -10., 5.]
        y = transform(x, self.R, self.t)
        for f in (estimate_svd, estimate_quaternion):
            self._assert_exact_fit(f, x, y)

    def test_both_same_noise_fit(self):
        noisy = self.Y + self.rng.normal(scale=0.02, size=self.Y.shape)
        r1, t1 = estimate_svd(self.X, noisy)
        r2, t2 = estimate_quaternion(self.X, noisy)
        np.testing.assert_allclose(r1, r2, atol=1e-10)
        np.testing.assert_allclose(t1, t2, atol=1e-10)

    def test_reflections_not_allowed(self):
        reflected = self.X @ np.diag([-1., 1., 1.])
        for f in (estimate_svd, estimate_quaternion):
            r, t = f(self.X, reflected)
            self.assertAlmostEqual(np.linalg.det(r), 1.0, places=10)

    def test_planar(self):
        x = self.X.copy()
        x[:, 2] = 0
        y = transform(x, self.R, self.t)
        for f in (estimate_svd, estimate_quaternion):
            self._assert_exact_fit(f, x, y)

    def test_input_validation(self):
        for f in (estimate_svd, estimate_quaternion):
            with self.assertRaises(ValueError):
                f(np.ones((2, 3)), np.ones((2, 3)))
            with self.assertRaises(ValueError):
                f(self.X, self.Y[:2])

    def test_geometry_ratio(self):
        x = self.rng.normal(size=(100, 3))
        x[:, 1:] *= 0.001
        self.assertLess(geometry_ratio(x), 0.01)

    def test_rotation_error(self):
        self.assertLess(rotation_error_degrees(self.R, self.R), 0.00001)
        self.assertAlmostEqual(rotation_error_degrees(axis_angle_to_rotation([0,0,1],90), np.eye(3)), 90, places=8)


if __name__ == '__main__':
    unittest.main()
