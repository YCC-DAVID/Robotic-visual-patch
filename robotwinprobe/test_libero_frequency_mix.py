"""Analytic signal and shape/range audits for CPU-only spatial-frequency exchange."""
import json
import unittest

import numpy as np

from libero_frequency_mix import blend, common_affine, decompose, lowpass, quantize_uint8, swap_metrics, variants


class FrequencyMixTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((64, 64))
        self.low = .12 * np.sin(2 * np.pi * 4 * x / 64)
        self.clean_high = .10 * np.sin(2 * np.pi * 16 * x / 64)
        self.attack_high = .07 * np.cos(2 * np.pi * 20 * y / 64)
        self.a = np.repeat((.5 + self.low + self.attack_high)[..., None], 3, axis=-1)
        self.b = np.repeat((.4 + self.clean_high)[..., None], 3, axis=-1)

    def test_analytic_four_controls_recover_the_intended_sinusoids(self):
        outputs = variants(self.a, self.b, 1 / 8)
        expected = dict(clean=self.b, full=self.a,
                        low_attack_high_clean=np.repeat(
                            (.5 + self.low + self.clean_high)[..., None], 3, axis=-1),
                        high_attack_low_clean=np.repeat(
                            (.4 + self.attack_high)[..., None], 3, axis=-1))
        for mode, (image, audit) in outputs.items():
            np.testing.assert_allclose(image, expected[mode], atol=2e-15)
            self.assertLess(audit['low_band_rmse'], 2e-15)
            self.assertLess(audit['high_band_rmse'], 2e-15)
            self.assertLess(audit['decomposition_reconstruction_max_error'], 2e-15)
            json.dumps(audit, allow_nan=False)

    def test_additive_difference_identity(self):
        low, _ = blend(self.a, self.b, 1 / 8, 'low_attack_high_clean')
        high, _ = blend(self.a, self.b, 1 / 8, 'high_attack_low_clean')
        np.testing.assert_allclose(low, self.b + lowpass(self.a - self.b, 1 / 8), atol=2e-15)
        np.testing.assert_allclose(low + high, self.a + self.b, atol=2e-15)

    def test_spatial_axes_preserve_layout_and_do_not_mix_batch_or_channels(self):
        hwc, _ = blend(self.a, self.b)
        chw, _ = blend(self.a.transpose(2, 0, 1), self.b.transpose(2, 0, 1),
                       spatial_axes=(-2, -1))
        np.testing.assert_allclose(chw.transpose(1, 2, 0), hwc, atol=2e-15)
        nhwc, _ = blend(np.stack([self.a, self.b]), np.stack([self.b, self.a]))
        nchw, _ = blend(np.stack([self.a, self.b]).transpose(0, 3, 1, 2),
                        np.stack([self.b, self.a]).transpose(0, 3, 1, 2), spatial_axes=(-2, -1))
        np.testing.assert_allclose(nchw.transpose(0, 2, 3, 1), nhwc, atol=2e-15)
        # Reversed axis order must also use the same spatial frequency plane.
        reversed_axes, _ = blend(self.a, self.b, spatial_axes=(1, 0))
        np.testing.assert_allclose(reversed_axes, hwc, atol=2e-15)

    def test_grayscale_odd_rectangular_images_and_complementary_projection(self):
        rng = np.random.default_rng(31)
        image = rng.normal(size=(63, 65))
        low, high = decompose(image, 1 / 8)
        np.testing.assert_allclose(low + high, image, atol=1e-15)
        np.testing.assert_allclose(lowpass(low, 1 / 8), low, atol=1e-15)
        np.testing.assert_allclose(lowpass(high, 1 / 8), 0, atol=1e-15)

    def test_cutoff_includes_boundary_and_dc(self):
        _, x = np.indices((64, 64))
        boundary = np.sin(2 * np.pi * 8 * x / 64)
        np.testing.assert_allclose(lowpass(boundary, 1 / 8), boundary, atol=4e-15)
        dc = lowpass(self.a, 0)
        np.testing.assert_allclose(dc, .5, atol=2e-15)
        np.testing.assert_allclose(lowpass(self.a, np.sqrt(.5)), self.a, atol=2e-15)

    def test_sources_are_not_mutated_and_alias_controls_are_independent(self):
        a, b = self.a.copy(), self.b.copy()
        a.setflags(write=False); b.setflags(write=False)
        outputs = variants(a, b)
        np.testing.assert_array_equal(a, self.a)
        np.testing.assert_array_equal(b, self.b)
        outputs['full'][0][...] = 0
        np.testing.assert_array_equal(a, self.a)

    def test_out_of_range_is_reported_without_silent_clipping(self):
        # Both sources are valid. The low attack DC plus clean fine detail
        # exceeds 1, which must remain visible to the caller before encoding.
        _, x = np.indices((64, 64))
        a = np.full((64, 64), .95)
        b = .5 + .4 * np.cos(2 * np.pi * 16 * x / 64)
        mixed, audit = blend(a, b)
        self.assertGreater(mixed.max(), 1)
        self.assertGreater(audit['out_of_range_fraction'], 0)
        self.assertFalse(audit['clipped'])
        encoded, quant = quantize_uint8(mixed)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertGreater(quant['clipping_rmse'], 0)
        decoded = encoded.astype(float) / 255
        after = swap_metrics(decoded, a, b, 1 / 8, 'low_attack_high_clean')
        self.assertGreater(after['high_band_rmse'], .01)
        self.assertGreater(after['low_band_rmse'], .01)

    def test_affine_rgb_normalization_commutes_with_frequency_exchange(self):
        raw, _ = blend(self.a, self.b)
        norm, _ = blend(2 * self.a - 1, 2 * self.b - 1, value_range=(-1, 1))
        np.testing.assert_allclose(norm, 2 * raw - 1, atol=2e-15)
        encoded, audit = quantize_uint8(norm, value_range=(-1, 1))
        expected = np.rint(np.clip(raw, 0, 1) * 255).astype(np.uint8)
        # Float-equivalent normalization paths can straddle half-integer
        # rounding ties. Such a tie may differ by one uint8 level, never more.
        self.assertLessEqual(np.max(np.abs(encoded.astype(int) - expected.astype(int))), 1)
        self.assertLessEqual(audit['quantization_only_rmse'], 1 / 255 + 1e-15)

    def test_common_affine_is_identity_for_legal_controls(self):
        controls = variants(self.a, self.b)
        transformed, audit = common_affine(controls)
        self.assertTrue(audit['identity'])
        for name in controls:
            np.testing.assert_array_equal(transformed[name][0], controls[name][0])
            self.assertEqual(transformed[name][1]['common_affine_scale'], 1)
        json.dumps(audit, allow_nan=False)

    def test_common_affine_avoids_clipping_and_preserves_exchanged_bands(self):
        _, x = np.indices((64, 64))
        a = np.full((64, 64), .95)
        b = .5 + .4 * np.cos(2 * np.pi * 16 * x / 64)
        controls = variants(a, b)
        mixed, audit = common_affine(controls)
        json.dumps(audit, allow_nan=False)
        self.assertIs(type(audit['scale']), float)
        self.assertIs(type(audit['offset']), float)
        self.assertIs(type(audit['identity']), bool)
        self.assertFalse(audit['identity'])
        self.assertGreater(audit['scale'], 0)
        self.assertLess(audit['scale'], 1)
        self.assertFalse(audit['clipped'])
        a_affine, b_affine = mixed['full'][0], mixed['clean'][0]
        for name, (image, _) in mixed.items():
            self.assertGreaterEqual(image.min(), 0)
            self.assertLessEqual(image.max(), 1)
            np.testing.assert_allclose(image, controls[name][0] * audit['scale'] + audit['offset'],
                                       atol=1e-15)
            metric = swap_metrics(image, a_affine, b_affine, 1 / 8, name)
            self.assertLess(metric['low_band_rmse'], 1e-15)
            self.assertLess(metric['high_band_rmse'], 1e-15)
            json.dumps(mixed[name][1], allow_nan=False)
        # The shared fit genuinely changes native clean/full; those need to be
        # retained separately by the caller for native positive controls.
        self.assertFalse(np.array_equal(a_affine, a))
        self.assertFalse(np.array_equal(b_affine, b))

    def test_common_affine_array_mapping_and_validation(self):
        arrays = {name: image for name, (image, _) in variants(self.a, self.b).items()}
        result, audit = common_affine(arrays, value_range=(-1, 1))
        self.assertTrue(audit['identity'])
        for name in arrays:
            self.assertIsInstance(result[name], np.ndarray)
            np.testing.assert_array_equal(result[name], arrays[name])
        with self.assertRaises(ValueError):common_affine({})
        with self.assertRaises(ValueError):common_affine({'a': np.zeros((2, 3)), 'b': np.zeros((3, 2))})

    def test_rejects_invalid_shapes_axes_cutoffs_and_values(self):
        for cutoff in (-1., float('nan'), float('inf'), 1.):
            with self.assertRaises(ValueError):blend(self.a, self.b, cutoff)
        for axes in ((0,), (0, 0), (0, 5)):
            with self.assertRaises(ValueError):blend(self.a, self.b, spatial_axes=axes)
        with self.assertRaises(ValueError):blend(self.a, self.b[:32])
        with self.assertRaises(ValueError):blend(self.a, self.b, mode='unknown')
        with self.assertRaises(ValueError):blend(self.a * np.nan, self.b)
        with self.assertRaises(ValueError):quantize_uint8(self.a, value_range=(1, 0))


if __name__ == '__main__':
    unittest.main()
