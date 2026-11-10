"""CPU-only spatial-frequency exchange for paired, same-state policy images.

This is a model-input diagnostic in encoded RGB (or the policy's affine RGB
normalization), not a physically realizable material or a camera MTF model.
Frequencies are radial cycles per image pixel, not cycles per physical patch.
The Fourier transform assumes periodic image boundaries. No ROI mask, window,
padding, clipping, or quantization is silently applied: each would change the
claimed frequency exchange. In particular, image-domain low-pass differences
can extend beyond the footprint of a physical patch.

Inputs are numeric arrays without automatic uint8 scaling. HWC / NHWC arrays
use the default spatial_axes=(-3, -2); CHW / NCHW use (-2, -1). Two-dimensional
grayscale arrays are supported as well. Outputs are float64 for auditing.

    mixed, audit = blend(attack, clean, 1 / 8,
                         'low_attack_high_clean', spatial_axes=(-2, -1),
                         value_range=(-1, 1))
    encoded, quant_audit = quantize_uint8(mixed, value_range=(-1, 1))

``blend`` never clips. ``quantize_uint8`` explicitly clips and reports its
error; to check the actual quantized swap, dequantize and call swap_metrics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


PRIMARY_CUTOFF = 1 / 8
SENSITIVITY_CUTOFFS = (1 / 16, 1 / 4)
MODES = ('clean', 'full', 'low_attack_high_clean', 'high_attack_low_clean')


def _array(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim < 2 or not np.issubdtype(source.dtype, np.number):
        raise ValueError('Expected a numeric image array with at least two dimensions')
    if np.iscomplexobj(source) or not np.isfinite(source).all():
        raise ValueError('Image values must be real and finite')
    return source.astype(np.float64, copy=True)


def _axes(ndim: int, spatial_axes: Sequence[int]) -> tuple[int, int]:
    if ndim == 2 and tuple(spatial_axes) == (-3, -2):
        return (0, 1)
    if len(spatial_axes) != 2:
        raise ValueError('Exactly two spatial axes are required')
    if any(not isinstance(a, (int, np.integer)) or not -ndim <= a < ndim
           for a in spatial_axes):
        raise ValueError('Spatial axes are out of range')
    axes = tuple(int(a) % ndim for a in spatial_axes)
    if axes[0] == axes[1]:
        raise ValueError('Spatial axes must differ')
    return axes


def _mask(shape: tuple[int, ...], axes: tuple[int, int], cutoff: float) -> np.ndarray:
    if not math.isfinite(cutoff) or not 0 <= cutoff <= math.sqrt(.5):
        raise ValueError('Radial cutoff must be in [0, sqrt(0.5)] cycles/pixel')
    radius_squared = 0.
    for axis in axes:
        if shape[axis] == 0:
            raise ValueError('Image spatial dimensions must be nonempty')
        broadcast = [1] * len(shape)
        broadcast[axis] = shape[axis]
        radius_squared = radius_squared + np.fft.fftfreq(shape[axis]).reshape(broadcast) ** 2
    return radius_squared <= float(cutoff) ** 2


def lowpass(image: np.ndarray, cutoff: float = PRIMARY_CUTOFF, *,
            spatial_axes: Sequence[int] = (-3, -2)) -> np.ndarray:
    """Orthogonal radial Fourier projection, including DC and the cutoff edge."""
    image = _array(image)
    axes = _axes(image.ndim, spatial_axes)
    mask = _mask(image.shape, axes, float(cutoff))
    return np.fft.ifftn(np.fft.fftn(image, axes=axes) * mask, axes=axes).real


def decompose(image: np.ndarray, cutoff: float = PRIMARY_CUTOFF, *,
              spatial_axes: Sequence[int] = (-3, -2)) -> tuple[np.ndarray, np.ndarray]:
    """Return complementary low/high components, whose sum reconstructs input."""
    image = _array(image)
    low = lowpass(image, cutoff, spatial_axes=spatial_axes)
    return low, image - low


def _bounds(value_range: Sequence[float]) -> tuple[float, float]:
    if len(value_range) != 2:
        raise ValueError('value_range must contain minimum and maximum')
    lower, upper = map(float, value_range)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError('value_range must have finite, increasing bounds')
    return lower, upper


def range_stats(image: np.ndarray, *, value_range: Sequence[float] = (0., 1.)) -> dict:
    image = _array(image)
    lower, upper = _bounds(value_range)
    if image.size == 0:
        raise ValueError('Image must be nonempty')
    below, above = image < lower, image > upper
    return dict(minimum=float(image.min()), maximum=float(image.max()),
                below_range_fraction=float(below.mean()), above_range_fraction=float(above.mean()),
                out_of_range_fraction=float((below | above).mean()),
                maximum_undershoot=max(0., lower - float(image.min())),
                maximum_overshoot=max(0., float(image.max()) - upper),
                value_range=[lower, upper])


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise ValueError(f'Paired image shapes differ: {a.shape} != {b.shape}')
    return a, b


def _rms(image: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(image))))


def swap_metrics(image: np.ndarray, a: np.ndarray, b: np.ndarray, cutoff: float,
                 mode: str, *, spatial_axes: Sequence[int] = (-3, -2)) -> dict:
    """Audit retained bands against their intended sources (a=attack, b=clean)."""
    if mode not in MODES:
        raise ValueError(f'Unknown frequency-exchange mode: {mode}')
    a, b = _paired(a, b)
    image = _array(image)
    if image.shape != a.shape:
        raise ValueError('Output and source shapes differ')
    axes = _axes(image.ndim, spatial_axes)
    la, ha = decompose(a, cutoff, spatial_axes=axes)
    lb, hb = decompose(b, cutoff, spatial_axes=axes)
    actual_low, actual_high = decompose(image, cutoff, spatial_axes=axes)
    desired_low = la if mode in ('full', 'low_attack_high_clean') else lb
    desired_high = ha if mode in ('full', 'high_attack_low_clean') else hb
    low_error, high_error = actual_low - desired_low, actual_high - desired_high
    return dict(low_band_rmse=_rms(low_error), high_band_rmse=_rms(high_error),
                low_band_max_error=float(np.max(np.abs(low_error))),
                high_band_max_error=float(np.max(np.abs(high_error))),
                low_delta_from_clean_rms=_rms(actual_low - lb),
                high_delta_from_clean_rms=_rms(actual_high - hb),
                attack_low_delta_rms=_rms(la - lb), attack_high_delta_rms=_rms(ha - hb),
                decomposition_reconstruction_max_error=float(np.max(np.abs(
                    actual_low + actual_high - image))))


def blend(a: np.ndarray, b: np.ndarray, cutoff: float = PRIMARY_CUTOFF,
          mode: str = 'low_attack_high_clean', *,
          spatial_axes: Sequence[int] = (-3, -2),
          value_range: Sequence[float] = (0., 1.)) -> tuple[np.ndarray, dict]:
    """Return float mixed image and JSON-serializable audit; never clip inputs/output.

    ``clean`` = L(b)+H(b); ``full`` = L(a)+H(a);
    ``low_attack_high_clean`` = L(a)+H(b) = b+L(a-b);
    ``high_attack_low_clean`` = L(b)+H(a) = b+H(a-b).
    """
    if mode not in MODES:
        raise ValueError(f'Unknown frequency-exchange mode: {mode}')
    a, b = _paired(a, b)
    axes = _axes(a.ndim, spatial_axes)
    la, ha = decompose(a, cutoff, spatial_axes=axes)
    lb, hb = decompose(b, cutoff, spatial_axes=axes)
    if mode == 'clean':
        result = b.copy()
    elif mode == 'full':
        result = a.copy()
    elif mode == 'low_attack_high_clean':
        result = la + hb
    else:
        result = lb + ha
    audit = dict(mode=mode, cutoff_cycles_per_pixel=float(cutoff), spatial_axes=list(axes),
                 dtype=str(result.dtype), shape=list(result.shape),
                 low_frequency_source='attack' if mode in ('full', 'low_attack_high_clean') else 'clean',
                 high_frequency_source='attack' if mode in ('full', 'high_attack_low_clean') else 'clean',
                 clipped=False, quantized=False, input_scaling='none',
                 scope='encoded-RGB/model-input spatial-frequency diagnostic; not physical render',
                 **range_stats(result, value_range=value_range),
                 **swap_metrics(result, a, b, cutoff, mode, spatial_axes=axes))
    return result, audit


def variants(a: np.ndarray, b: np.ndarray, cutoff: float = PRIMARY_CUTOFF, *,
             spatial_axes: Sequence[int] = (-3, -2),
             value_range: Sequence[float] = (0., 1.)) -> dict[str, tuple[np.ndarray, dict]]:
    """All four controls, with independent arrays and audit dictionaries."""
    return {mode: blend(a, b, cutoff, mode, spatial_axes=spatial_axes, value_range=value_range)
            for mode in MODES}


def common_affine(variants_mapping: Mapping, *,
                  value_range: Sequence[float] = (0., 1.)) -> tuple[dict, dict]:
    """Fit all four float controls with one positive affine map per camera.

    Input values can be arrays or ``(array, audit)`` pairs returned by variants;
    the output preserves that form. All variants must have the same shape.
    Already legal images get the identity map. Otherwise, the shared range is
    extended to include the requested endpoints and all source values, then
    compressed into the legal range. There is no clipping or separate per-mode
    normalization. An identical scalar map preserves complementary frequency
    exchange (with all source images transformed by that same map).

    This can change even the clean/full controls; retain native clean/full as
    separate positive controls. Call this once for each camera, not for each
    condition. The function does not perform model-dtype casting.
    """
    if not variants_mapping:
        raise ValueError('At least one image variant is required')
    lower, upper = _bounds(value_range)
    images, audits = {}, {}
    for name, value in variants_mapping.items():
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            images[name], audits[name] = _array(value[0]), dict(value[1])
        else:
            images[name] = _array(value)
    shapes = {image.shape for image in images.values()}
    if len(shapes) != 1 or any(image.size == 0 for image in images.values()):
        raise ValueError('All variants must have one common nonempty shape')
    minimum = min(float(image.min()) for image in images.values())
    maximum = max(float(image.max()) for image in images.values())
    if minimum >= lower and maximum <= upper:
        scale, offset = 1., 0.
    else:
        fit_lower, fit_upper = min(lower, minimum), max(upper, maximum)
        # A minute interior margin prevents arithmetic roundoff from producing
        # illegal endpoints while preserving a single strictly affine map.
        margin = 32 * np.finfo(np.float64).eps * (upper - lower)
        scale = (upper - lower - 2 * margin) / (fit_upper - fit_lower)
        offset = lower + margin - scale * fit_lower
    # NumPy's epsilon above can promote the nonidentity branch to np.float64
    # and its equality tests to np.bool_. Audit data must remain plain JSON.
    scale, offset = float(scale), float(offset)
    result = {}
    for name, image in images.items():
        transformed = image * scale + offset
        if name in audits:
            audit = audits[name]
            audit.update(common_affine_scale=scale, common_affine_offset=offset,
                         common_affine_input_audit=dict(audits[name]))
            audit.update(range_stats(transformed, value_range=value_range))
            result[name] = (transformed, audit)
        else:
            result[name] = transformed
    if 'full' in result and 'clean' in result:
        full = result['full'][0] if 'full' in audits else result['full']
        clean = result['clean'][0] if 'clean' in audits else result['clean']
        for name in audits:
            audit = result[name][1]
            if name in MODES and 'cutoff_cycles_per_pixel' in audit:
                audit.update(swap_metrics(result[name][0], full, clean,
                                          audit['cutoff_cycles_per_pixel'], name,
                                          spatial_axes=audit['spatial_axes']))
                audit['frequency_audit_scope'] = 'post-affine against equally transformed clean/full'
    else:
        for name in audits:
            result[name][1]['frequency_audit_scope'] = 'pre-affine; source controls unavailable for a new audit'
    meta = dict(scale=scale, offset=offset, identity=bool(scale == 1. and offset == 0.),
                input_minimum=minimum, input_maximum=maximum, value_range=[lower, upper],
                modes=list(images), clipped=False, per_variant_normalization=False,
                scope='shared positive scalar affine map across all controls for one camera',
                native_clean_full_preserved=False,
                output_ranges={name: range_stats(image * scale + offset, value_range=value_range)
                               for name, image in images.items()})
    return result, meta


def quantize_uint8(image: np.ndarray, *,
                   value_range: Sequence[float] = (0., 1.)) -> tuple[np.ndarray, dict]:
    """Explicitly clip/quantize to native policy RGB, reporting both error sources.

    For a spectral audit, reconstruct float using
    ``encoded.astype(float)/255*(upper-lower)+lower`` then call swap_metrics.
    Do not apply this conversion to already preprocessed model tensors unless
    intentionally testing the additional clipping/quantization step.
    """
    image = _array(image)
    lower, upper = _bounds(value_range)
    clipped = np.clip(image, lower, upper)
    encoded = np.rint((clipped - lower) / (upper - lower) * 255).astype(np.uint8)
    decoded = encoded.astype(np.float64) / 255 * (upper - lower) + lower
    audit = dict(**range_stats(image, value_range=value_range),
                 clipping_applied=True, quantization_applied=True,
                 clipping_rmse=_rms(clipped - image),
                 quantization_only_rmse=_rms(decoded - clipped),
                 total_rmse=_rms(decoded - image),
                 total_max_error=float(np.max(np.abs(decoded - image))),
                 warning='Clipping/uint8 quantization can introduce frequencies outside the intended exchanged band')
    return encoded, audit
