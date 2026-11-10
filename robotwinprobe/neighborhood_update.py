"""CPU primitives for a one-step texture-neighborhood attack.

``make_neighbor`` takes an approximate *descent* step for the inner problem.
``ascent`` takes the outer ascent step from the original center, not from that
neighbor. Neither function claims that its finite step improves an objective.
The caller must evaluate losses and keep gradient-call accounting separately.

Textures use the existing PNG-oriented float32 (1, 3, 128, 128) convention.
Inputs are never modified. Radius checks use float64 differences of the actual
float32 outputs, including at texture bounds and float32 rounding boundaries.
"""

from numbers import Real

import numpy as np


TEXTURE_SHAPE = (1, 3, 128, 128)


def _texture(value, name):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != TEXTURE_SHAPE:
        raise ValueError(f"{name} must have shape {TEXTURE_SHAPE}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _gradient(value):
    if not isinstance(value, np.ndarray):
        raise TypeError("gradient must be a numpy array")
    if value.shape != TEXTURE_SHAPE:
        raise ValueError(f"gradient must have shape {TEXTURE_SHAPE}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("gradient must have a floating dtype")
    if not np.isfinite(value).all():
        raise ValueError("gradient must be finite")
    return value


def _scalar(value, name, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be no greater than {upper}")
    return result


def _bounded_sign_step(q, gradient, distance, direction):
    # Compute in float64 first. A nearest float32 rounding can otherwise move
    # farther than the requested L-infinity radius by a fraction of an ulp.
    center = q.astype(np.float64)
    signs = np.sign(gradient).astype(np.float64)
    target = np.clip(center + direction * distance * signs, 0, 1)
    result = target.astype(np.float32)
    overshot = np.abs(result.astype(np.float64) - center) > distance
    if np.any(overshot):
        result[overshot] = np.nextafter(result[overshot], q[overshot])
    # This is a runtime invariant on emitted float32 values, not a tolerance.
    if np.any(np.abs(result.astype(np.float64) - center) > distance):
        raise ArithmeticError("float32 sign step exceeds requested radius")
    return result


def make_neighbor(q, gradient, rho):
    """Return clip(q - rho * sign(gradient), 0, 1), within actual radius rho.

    ``rho`` is a finite real scalar in [0, 1]. At a float32 rounding boundary,
    the returned value is rounded toward q if necessary to preserve the radius.
    Thus zero/sub-ulp radii can leave some or all pixels unchanged.
    """
    _texture(q, "q")
    _gradient(gradient)
    rho = _scalar(rho, "rho", upper=1.0)
    return _bounded_sign_step(q, gradient, rho, -1.0)


def ascent(q, gradient, lr):
    """Return an outer sign-ascent step from the supplied original center q.

    ``gradient`` should be measured at the inner neighbor, while q remains the
    original center. ``lr`` is finite and nonnegative; zero is the identity.
    """
    _texture(q, "q")
    _gradient(gradient)
    lr = _scalar(lr, "lr")
    return _bounded_sign_step(q, gradient, lr, 1.0)


def describe_move(q, result):
    """Return JSON-serializable diagnostics of the actual stored movement."""
    _texture(q, "q")
    _texture(result, "result")
    difference = result.astype(np.float64) - q.astype(np.float64)
    return {
        "linf": float(np.abs(difference).max()),
        "l2": float(np.linalg.norm(difference.ravel())),
        "changed_elements": int(np.count_nonzero(difference)),
        "total_elements": int(difference.size),
        "minimum": float(result.min()),
        "maximum": float(result.max()),
        "at_bounds_elements": int(np.count_nonzero((result == 0) | (result == 1))),
    }
