from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger("mask_auto_enhancement")


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8

UINT8_MAX = 255.0
UINT16_MAX = 65535.0

# Used when calculating statistics.
DEFAULT_STAT_SAMPLE_SIZE = 250_000

# Maximum recommended analysis dimension.
# The actual image pipeline may use a different value.
DEFAULT_MAX_ANALYSIS_DIMENSION = 1800


# ============================================================
# NUMERIC UTILITIES
# ============================================================

def clip(
    value: float,
    low: float,
    high: float,
) -> float:
    """Safely clamp a scalar value."""

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


def clip_array(
    value: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Safely clamp an array without changing its dtype unnecessarily."""

    return np.clip(
        value,
        low,
        high,
    )


def safe_ratio(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Numerically safe ratio.

    Unlike the previous implementation this preserves
    the sign of the denominator.
    """

    if not np.isfinite(numerator):
        return float(default)

    if not np.isfinite(denominator):
        return float(default)

    if abs(denominator) <= EPS:
        return float(default)

    return float(
        numerator / denominator
    )


def finite_values(
    values: np.ndarray,
) -> np.ndarray:
    """Return finite values only."""

    values = np.asarray(values)

    if values.size == 0:
        return values

    return values[
        np.isfinite(values)
    ]


# ============================================================
# ROBUST STATISTICS
# ============================================================

def robust_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    """
    Percentile calculation that safely ignores NaN/Inf.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = finite_values(values)

    if values.size == 0:
        return 0.0

    percentile = float(
        np.clip(
            percentile,
            0.0,
            100.0,
        )
    )

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def robust_mean(
    values: np.ndarray,
    low: float = 5.0,
    high: float = 95.0,
) -> float:
    """
    Trimmed mean.

    More stable than a simple mean for image statistics because
    highlights, shadows, segmentation errors and isolated pixels
    cannot dominate the result.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = finite_values(values)

    if values.size == 0:
        return 0.0

    low = float(
        np.clip(
            low,
            0.0,
            100.0,
        )
    )

    high = float(
        np.clip(
            high,
            low,
            100.0,
        )
    )

    p_low = np.percentile(
        values,
        low,
    )

    p_high = np.percentile(
        values,
        high,
    )

    selected = values[
        (values >= p_low)
        &
        (values <= p_high)
    ]

    if selected.size == 0:
        return float(
            np.mean(values)
        )

    return float(
        np.mean(selected)
    )


def weighted_mean(
    values: np.ndarray,
    weights: Optional[np.ndarray],
) -> float:
    """
    Weighted mean for soft masks.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if weights is None:
        return robust_mean(values)

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if values.shape != weights.shape:
        raise ValueError(
            "values and weights must have identical shapes."
        )

    valid = (
        np.isfinite(values)
        &
        np.isfinite(weights)
        &
        (weights > 0.0)
    )

    if not np.any(valid):
        return 0.0

    values = values[valid]
    weights = weights[valid]

    total_weight = float(
        np.sum(weights)
    )

    if total_weight <= EPS:
        return 0.0

    return float(
        np.sum(values * weights)
        /
        total_weight
    )


def weighted_percentile(
    values: np.ndarray,
    weights: Optional[np.ndarray],
    percentile: float,
) -> float:
    """
    Weighted percentile with linear interpolation.

    This is more stable than simply returning the first value
    whose cumulative weight crosses the target.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return 0.0

    if weights is None:
        return robust_percentile(
            values,
            percentile,
        )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if values.shape != weights.shape:
        raise ValueError(
            "values and weights must have identical shapes."
        )

    valid = (
        np.isfinite(values)
        &
        np.isfinite(weights)
        &
        (weights > 0.0)
    )

    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return 0.0

    order = np.argsort(
        values,
        kind="mergesort",
    )

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(
        weights,
        dtype=np.float64,
    )

    total_weight = float(
        cumulative[-1]
    )

    if total_weight <= EPS:
        return 0.0

    percentile = float(
        np.clip(
            percentile,
            0.0,
            100.0,
        )
    )

    # Use the centre of each weight interval.
    positions = (
        cumulative
        -
        0.5 * weights
    )

    target = (
        percentile / 100.0
    ) * total_weight

    if target <= positions[0]:
        return float(values[0])

    if target >= positions[-1]:
        return float(values[-1])

    index = int(
        np.searchsorted(
            positions,
            target,
            side="right",
        )
    )

    index = min(
        max(index, 1),
        len(values) - 1,
    )

    x0 = positions[index - 1]
    x1 = positions[index]

    y0 = values[index - 1]
    y1 = values[index]

    if abs(x1 - x0) <= EPS:
        return float(y0)

    fraction = (
        target - x0
    ) / (
        x1 - x0
    )

    return float(
        y0 + fraction * (y1 - y0)
    )


# ============================================================
# STATISTICAL SAMPLING
# ============================================================

def sample_values(
    values: np.ndarray,
    max_samples: int = DEFAULT_STAT_SAMPLE_SIZE,
    seed: int = 17,
) -> np.ndarray:
    """
    Deterministically subsample a large array.

    This prevents expensive percentile calculations over
    millions of pixels while keeping analysis reproducible.
    """

    values = np.asarray(values)

    if values.size <= max_samples:
        return values

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be greater than zero."
        )

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        values.size,
        size=max_samples,
        replace=False,
    )

    return values.reshape(-1)[indices]


def sample_masked_values(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_samples: int = DEFAULT_STAT_SAMPLE_SIZE,
    seed: int = 17,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract values and optional weights from a mask and
    deterministically sample them.
    """

    values = np.asarray(values)

    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != values.shape:
            raise ValueError(
                "mask and values must have identical shapes."
            )

        valid = (
            mask > 0
        )

        values = values[valid]

        if weights is not None:
            weights = np.asarray(weights)[valid]

    elif weights is not None:
        weights = np.asarray(weights)

        if weights.shape != values.shape:
            raise ValueError(
                "weights and values must have identical shapes."
            )

    if values.size == 0:
        return (
            values,
            weights,
        )

    if values.size > max_samples:
        rng = np.random.default_rng(seed)

        indices = rng.choice(
            values.size,
            size=max_samples,
            replace=False,
        )

        values = values[indices]

        if weights is not None:
            weights = weights[indices]

    return (
        values,
        weights,
    )


# ============================================================
# COLOR SCIENCE
# ============================================================

def srgb_to_linear(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized sRGB to linear RGB.

    Input:
        float32/float64 in [0, 1]

    Output:
        float32 in [0, 1]
    """

    x = np.asarray(
        image,
        dtype=np.float32,
    )

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    result = np.empty_like(x)

    low = x <= 0.04045

    result[low] = (
        x[low]
        /
        12.92
    )

    high = ~low

    result[high] = np.power(
        (
            x[high] + 0.055
        )
        /
        1.055,
        2.4,
    )

    return result


def linear_to_srgb(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert linear RGB to normalized sRGB.
    """

    x = np.asarray(
        image,
        dtype=np.float32,
    )

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    result = np.empty_like(x)

    low = x <= 0.0031308

    result[low] = (
        x[low]
        *
        12.92
    )

    high = ~low

    result[high] = (
        1.055
        *
        np.power(
            x[high],
            1.0 / 2.4,
        )
        -
        0.055
    )

    return result


def bgr_to_linear_rgb(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Convert BGR uint8 image to linear RGB float32.
    """

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    rgb = (
        rgb.astype(
            np.float32,
            copy=False,
        )
        /
        UINT8_MAX
    )

    return srgb_to_linear(
        rgb
    )


def luminance_from_linear_rgb(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Rec.709 luminance in linear light.
    """

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            "RGB array must have shape HxWx3."
        )

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    ).astype(
        np.float32,
        copy=False,
    )


def linear_rgb_to_lab(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert linear RGB to CIE Lab using D65.

    RGB input is assumed to be linear RGB in [0, 1].
    """

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            "RGB array must have shape HxWx3."
        )

    rgb = np.asarray(
        rgb,
        dtype=np.float32,
    )

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    x = (
        0.4124564 * r
        +
        0.3575761 * g
        +
        0.1804375 * b
    )

    y = (
        0.2126729 * r
        +
        0.7151522 * g
        +
        0.0721750 * b
    )

    z = (
        0.0193339 * r
        +
        0.1191920 * g
        +
        0.9503041 * b
    )

    # D65 reference white.
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f(
        value: np.ndarray,
    ) -> np.ndarray:

        value = np.maximum(
            value,
            0.0,
        )

        result = np.empty_like(
            value,
            dtype=np.float32,
        )

        high = value > delta3

        result[high] = np.cbrt(
            value[high]
        )

        result[~high] = (
            value[~high]
            /
            (
                3.0
                *
                delta
                *
                delta
            )
            +
            4.0 / 29.0
        )

        return result

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = (
        116.0 * fy
        -
        16.0
    )

    a = (
        500.0
        *
        (fx - fy)
    )

    b_value = (
        200.0
        *
        (fy - fz)
    )

    return np.stack(
        [
            L,
            a,
            b_value,
        ],
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_bgr_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Validate an image before entering the analysis pipeline.
    """

    if image is None:
        raise ValueError(
            "Image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "Image must be numpy.ndarray."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape HxWx3."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected BGR image with 3 channels."
        )

    height, width = image.shape[:2]

    if height < 2 or width < 2:
        raise ValueError(
            "Image is too small for analysis."
        )

    if image.dtype not in (
        np.uint8,
        np.uint16,
        np.float16,
        np.float32,
        np.float64,
    ):
        raise TypeError(
            f"Unsupported image dtype: {image.dtype}"
        )

    if not np.all(
        np.isfinite(
            image
        )
    ):
        raise ValueError(
            "Image contains NaN or infinite values."
        )

    return image


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_bgr_to_float(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert supported BGR image formats to float32 [0,1].

    uint8:
        0..255

    uint16:
        0..65535

    floating point:
        Assumed to already be 0..1 unless values > 1 are detected.
    """

    image = validate_bgr_image(
        image
    )

    if image.dtype == np.uint8:

        result = (
            image.astype(
                np.float32,
                copy=False,
            )
            /
            UINT8_MAX
        )

    elif image.dtype == np.uint16:

        result = (
            image.astype(
                np.float32,
                copy=False,
            )
            /
            UINT16_MAX
        )

    else:

        result = image.astype(
            np.float32,
            copy=False,
        )

        max_value = float(
            np.max(result)
        )

        if max_value > 1.0 + 1e-6:

            # Common case for float images stored
            # in 0..255 range.
            if max_value <= 255.0 + 1e-3:

                result = (
                    result
                    /
                    UINT8_MAX
                )

            else:
                raise ValueError(
                    "Floating-point image values must be "
                    "in [0,1] or [0,255]."
                )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# COLOR DATA
# ============================================================

@dataclass
class ImageColorData:
    """
    Cached color representations used by the enhancement pipeline.

    Important:
        The arrays are analysis-resolution arrays.

    linear_rgb:
        Linear RGB, [0,1]

    luminance:
        Linear Rec.709 luminance

    lab:
        CIE Lab, D65

    hsv:
        OpenCV HSV using float representation:
            H = 0..360
            S = 0..1
            V = 0..1

    bgr_float:
        sRGB BGR, [0,1]
    """

    linear_rgb: np.ndarray
    luminance: np.ndarray
    lab: np.ndarray

    # Optional because HSV is not required by every action.
    hsv: Optional[np.ndarray] = None

    # Kept for compatibility with existing code.
    bgr_float: Optional[np.ndarray] = None


# ============================================================
# COLOR DATA PREPARATION
# ============================================================

def prepare_color_data(
    image: np.ndarray,
    *,
    include_hsv: bool = True,
    include_bgr_float: bool = True,
) -> ImageColorData:
    """
    Prepare cached color representations.

    Parameters
    ----------
    image:
        BGR image.

    include_hsv:
        Generate HSV only when required.

    include_bgr_float:
        Keep normalized BGR representation.

    Notes
    -----
    The image should already be resized to analysis resolution
    before calling this function.

    Do not run this directly on a 6000x4000 image unless the
    additional memory usage is intentional.
    """

    image = validate_bgr_image(
        image
    )

    bgr_float = normalize_bgr_to_float(
        image
    )

    # Convert normalized BGR -> RGB.
    rgb = cv2.cvtColor(
        bgr_float,
        cv2.COLOR_BGR2RGB,
    )

    linear_rgb = srgb_to_linear(
        rgb
    )

    luminance = luminance_from_linear_rgb(
        linear_rgb
    )

    lab = linear_rgb_to_lab(
        linear_rgb
    )

    hsv = None

    if include_hsv:

        hsv = cv2.cvtColor(
            bgr_float,
            cv2.COLOR_BGR2HSV,
        )

    return ImageColorData(
        linear_rgb=linear_rgb,
        luminance=luminance,
        lab=lab,
        hsv=hsv,
        bgr_float=(
            bgr_float
            if include_bgr_float
            else None
        ),
    )


# ============================================================
# MASK UTILITIES
# ============================================================

def validate_mask(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    name: str = "mask",
) -> np.ndarray:
    """
    Validate a hard or soft mask.
    """

    mask = np.asarray(
        mask
    )

    if mask.shape != image_shape:
        raise ValueError(
            f"{name} shape {mask.shape} does not match "
            f"image shape {image_shape}."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array."
        )

    if not np.all(
        np.isfinite(mask)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return np.clip(
        mask.astype(
            np.float32,
            copy=False,
        ),
        0.0,
        1.0,
    )


def mask_coverage(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate fraction of image covered by a mask.
    """

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if mask.size == 0:
        return 0.0

    return float(
        np.mean(
            mask >= threshold
        )
    )


def mask_weight(
    mask: np.ndarray,
    *,
    minimum_confidence: float = 0.05,
) -> np.ndarray:
    """
    Convert a soft mask into usable statistical weights.

    Very low-confidence segmentation pixels are discarded.
    """

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    result = np.where(
        mask >= minimum_confidence,
        mask,
        0.0,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# ROBUST IMAGE STATISTICS
# ============================================================

def luminance_statistics(
    luminance: np.ndarray,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_samples: int = DEFAULT_STAT_SAMPLE_SIZE,
) -> dict:
    """
    Calculate robust luminance statistics.

    This is useful for exposure, shadows, highlights,
    subject brightness and mask-aware tone correction.
    """

    values = np.asarray(
        luminance,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise ValueError(
            "luminance must be a 2D array."
        )

    values, weights = sample_masked_values(
        values,
        mask=mask,
        weights=weights,
        max_samples=max_samples,
    )

    values = finite_values(
        values
    )

    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "count": int(values.size),

        "mean": weighted_mean(
            values,
            weights,
        ),

        "median": weighted_percentile(
            values,
            weights,
            50.0,
        ),

        "p01": weighted_percentile(
            values,
            weights,
            1.0,
        ),

        "p05": weighted_percentile(
            values,
            weights,
            5.0,
        ),

        "p25": weighted_percentile(
            values,
            weights,
            25.0,
        ),

        "p75": weighted_percentile(
            values,
            weights,
            75.0,
        ),

        "p95": weighted_percentile(
            values,
            weights,
            95.0,
        ),

        "p99": weighted_percentile(
            values,
            weights,
            99.0,
        ),
    }


# ============================================================
# COLOR STATISTICS
# ============================================================

def lab_statistics(
    lab: np.ndarray,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_samples: int = DEFAULT_STAT_SAMPLE_SIZE,
) -> dict:
    """
    Robust Lab statistics.

    Particularly useful for:
        - neutral detection
        - skin analysis
        - color cast detection
        - saturation estimation
    """

    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(
            "Lab image must have shape HxWx3."
        )

    result = {}

    names = (
        "L",
        "a",
        "b",
    )

    for index, name in enumerate(names):

        channel = lab[..., index]

        values, channel_weights = sample_masked_values(
            channel,
            mask=mask,
            weights=weights,
            max_samples=max_samples,
            seed=17 + index,
        )

        result[name] = {
            "mean": weighted_mean(
                values,
                channel_weights,
            ),

            "median": weighted_percentile(
                values,
                channel_weights,
                50.0,
            ),

            "p05": weighted_percentile(
                values,
                channel_weights,
                5.0,
            ),

            "p95": weighted_percentile(
                values,
                channel_weights,
                95.0,
            ),
        }

    return result