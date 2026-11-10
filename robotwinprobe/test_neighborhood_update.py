"""Analytic CPU checks for neighborhood descent and centered outer ascent."""

import unittest

import numpy as np

from neighborhood_update import TEXTURE_SHAPE, ascent, describe_move, make_neighbor


def texture(value):
    return np.full(TEXTURE_SHAPE, value, dtype=np.float32)


class NeighborhoodUpdateTest(unittest.TestCase):
    def test_linear_inner_descent_and_centered_outer_step(self):
        q = texture(0.5)
        weights = np.ones(TEXTURE_SHAPE, dtype=np.float64)
        weights[:, 1] = -2
        weights[:, 2] = 0
        loss = lambda p: float(np.sum(p.astype(np.float64) * weights))
        neighbor = make_neighbor(q, weights, 0.1)
        self.assertLess(loss(neighbor), loss(q))
        updated = ascent(q, weights, 0.02)
        self.assertGreater(loss(updated), loss(q))
        self.assertGreater(float(updated[0, 0, 0, 0]), 0.5)
        # Updating from the inner point would remain below 0.5 in this example.
        wrong = ascent(neighbor, weights, 0.02)
        self.assertLess(float(wrong[0, 0, 0, 0]), 0.5)
        np.testing.assert_array_equal(updated[:, 2], q[:, 2])

    def test_quadratic_uses_gradient_at_neighbor_and_keeps_center(self):
        q = texture(0.55)
        minimum = texture(0.5)
        loss = lambda p: float(np.square(p.astype(np.float64) - minimum).sum())
        gradient = lambda p: 2 * (p.astype(np.float64) - minimum)
        # The inner step crosses the minimum; its loss still decreases and its
        # gradient changes sign. This distinguishes inner and outer gradients.
        neighbor = make_neighbor(q, gradient(q), 0.08)
        self.assertLess(loss(neighbor), loss(q))
        self.assertLess(float(gradient(neighbor).flat[0]), 0)
        updated = ascent(q, gradient(neighbor), 0.01)
        self.assertLess(float(updated.flat[0]), float(q.flat[0]))
        self.assertGreater(float(updated.flat[0]), float(neighbor.flat[0]))
        wrong_gradient = ascent(q, gradient(q), 0.01)
        self.assertGreater(float(wrong_gradient.flat[0]), float(q.flat[0]))

    def test_clipping_radius_and_input_immutability(self):
        rng = np.random.default_rng(7)
        q = rng.uniform(size=TEXTURE_SHAPE).astype(np.float32)
        q.flat[:4] = [0, 1, 0.001, 0.999]
        gradient = rng.standard_normal(TEXTURE_SHAPE).astype(np.float32)
        gradient.flat[:4] = [1, -1, 1, -1]
        before, grad_before = q.copy(), gradient.copy()
        neighbor = make_neighbor(q, gradient, 2 / 255)
        np.testing.assert_array_equal(q, before)
        np.testing.assert_array_equal(gradient, grad_before)
        self.assertIsNot(q, neighbor)
        self.assertEqual(neighbor.dtype, np.float32)
        self.assertGreaterEqual(float(neighbor.min()), 0)
        self.assertLessEqual(float(neighbor.max()), 1)
        self.assertEqual(float(neighbor.flat[0]), 0)
        self.assertEqual(float(neighbor.flat[1]), 1)
        report = describe_move(q, neighbor)
        self.assertLessEqual(report["linf"], 2 / 255)
        self.assertEqual(report["total_elements"], int(np.prod(TEXTURE_SHAPE)))
        self.assertGreater(report["changed_elements"], 0)

    def test_zero_radius_reduces_to_standard_sign_ascent(self):
        q = texture(0.5)
        gradient = texture(0.25)
        neighbor = make_neighbor(q, gradient, 0)
        np.testing.assert_array_equal(q, neighbor)
        self.assertIsNot(q, neighbor)
        np.testing.assert_array_equal(ascent(q, gradient, 0), q)
        np.testing.assert_array_equal(ascent(q, gradient, 0.125), texture(0.625))

    def test_sub_ulp_radius_and_nearest_rounding_are_feasible(self):
        q = texture(0.5)
        gradient = texture(1)
        # At q=.5, the next lower float32 is 2**-25 away. Nearest rounding of
        # this step would exceed rho, so the feasible stored result is q.
        rho = 0.75 * 2**-25
        neighbor = make_neighbor(q, gradient, rho)
        np.testing.assert_array_equal(neighbor, q)
        self.assertEqual(describe_move(q, neighbor)["linf"], 0)
        moved = make_neighbor(q, gradient, 2**-25)
        self.assertEqual(describe_move(q, moved)["linf"], 2**-25)
        self.assertTrue(np.all(moved < q))
        # A familiar decimal radius also has a strict actual-value bound.
        self.assertLessEqual(describe_move(q, make_neighbor(q, gradient, .1))["linf"], .1)

    def test_nonfinite_and_invalid_inputs_are_rejected(self):
        q, gradient = texture(0.5), texture(1)
        for value in [float("nan"), float("inf"), -.01, 1.01]:
            with self.subTest(rho=value), self.assertRaises(ValueError):
                make_neighbor(q, gradient, value)
        for value in [True, "0.1", np.array(.1), [0.1]]:
            with self.subTest(rho=value), self.assertRaises(TypeError):
                make_neighbor(q, gradient, value)
        for value in [float("nan"), float("inf"), -.1]:
            with self.subTest(lr=value), self.assertRaises(ValueError):
                ascent(q, gradient, value)
        for bad in [float("nan"), float("inf")]:
            broken = gradient.copy()
            broken.flat[0] = bad
            with self.assertRaises(ValueError):
                make_neighbor(q, broken, .1)
            broken = q.copy()
            broken.flat[0] = bad
            with self.assertRaises(ValueError):
                ascent(broken, gradient, .1)
        for bad in [-.01, 1.01]:
            broken = q.copy()
            broken.flat[0] = bad
            with self.assertRaises(ValueError):
                make_neighbor(broken, gradient, .1)
        with self.assertRaises(TypeError):
            make_neighbor(q.astype(np.float64), gradient, .1)
        with self.assertRaises(ValueError):
            make_neighbor(q[0], gradient, .1)
        with self.assertRaises(TypeError):
            make_neighbor(q, gradient.astype(np.int32), .1)
        with self.assertRaises(ValueError):
            make_neighbor(q, gradient[0], .1)


if __name__ == "__main__":
    unittest.main()
