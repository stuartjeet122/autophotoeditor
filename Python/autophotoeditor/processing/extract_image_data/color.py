"""
AutoPhotoEditor - Image Analysis V4
===================================

Color / Detail / Histogram / Grid Analysis
-------------------------------------------

This module produces analysis data for the automatic enhancement
pipeline. It does NOT modify the image.

Design goals
------------

1. Exposure-aware color analysis.
2. Skin-aware color analysis.
3. Robust saturation statistics.
4. Vibrance recommendations that avoid oversaturation.
5. Noise-aware detail analysis.
6. Resolution-independent detail estimation.
7. Robust histogram generation.
8. Spatial luminance analysis.
9. Grid clipping and contrast information.
10. Deterministic processing.
11. Backwards-compatible JSON fields.

Input
-----
BGR uint8 image.

Output
------
JSON-compatible dictionaries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from .shared import *


# ============================================================================
# CONSTANTS
# ============================================================================

COLOR_ANALYSIS_MAX_DIM = 1200
COLOR_SAMPLE_PIXELS = 120_000
SKIN_SAMPLE_PIXELS = 30_000

DETAIL_MAX_DIM = 1600
DETAIL_SAMPLE_PIXELS = 120_000

HISTOGRAM_SAMPLE_PIXELS = 150_000

GRID_SAMPLE_PIXELS = 20_000

EPS = 1e-8


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _deterministic_sample(
    values: np.ndarray,
    max_items: int,
) -> np.ndarray:
    """
    Deterministically sample an array.

    Important:
        This helper flattens multidimensional arrays.

    It is suitable for scalar analysis arrays such as:
        luminance
        saturation
        gradients
        histograms

    It should NOT be used on RGB/BGR arrays when the channel
    structure must be preserved.
    """

    arr = np.asarray(values)

    if arr.size == 0:
        return arr

    if max_items <= 0:
        return arr[:0]

    if arr.ndim == 0:
        return arr.reshape(1)

    arr = arr.reshape(-1)

    if arr.size <= max_items:
        return arr

    indices = np.linspace(
        0,
        arr.size - 1,
        max_items,
        dtype=np.int64,
    )

    return arr[
        indices
    ]


def _deterministic_indices(
    size: int,
    max_items: int,
) -> np.ndarray:
    """
    Generate deterministic flattened indices.

    This is important when multiple arrays must remain paired.

    Example:

        saturation[index]
        luminance[index]
        value[index]

    all refer to the same pixels.
    """

    if size <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if max_items <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if size <= max_items:
        return np.arange(
            size,
            dtype=np.int64,
        )

    return np.linspace(
        0,
        size - 1,
        max_items,
        dtype=np.int64,
    )


def _resize_for_analysis(
    image: np.ndarray,
    max_dim: int,
) -> np.ndarray:
    """
    Resize only when necessary.

    INTER_AREA is used for downsampling because it provides
    stable deterministic statistical results.
    """

    if image is None:
        raise ValueError(
            "image is None."
        )

    if image.ndim < 2:
        raise ValueError(
            "image must have at least two dimensions."
        )

    h, w = image.shape[:2]

    largest = max(
        h,
        w,
    )

    if largest <= max_dim:
        return image

    scale = (
        float(max_dim)
        / float(largest)
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    return cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def _safe_percentile(
    values: np.ndarray,
    q: float,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    values = np.asarray(
        values
    )

    if values.size == 0:
        return default

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return default

    return float(
        np.percentile(
            values,
            q,
        )
    )


def _safe_mean(
    values: np.ndarray,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    values = np.asarray(
        values
    )

    if values.size == 0:
        return default

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return default

    return float(
        np.mean(
            values
        )
    )


def _safe_fraction(
    condition: np.ndarray,
) -> float:

    condition = np.asarray(
        condition
    )

    if condition.size == 0:
        return 0.0

    return float(
        np.mean(
            condition
        )
    )


def _prepare_analysis_mask(
    mask: Optional[np.ndarray],
    target_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Convert a semantic mask into a boolean mask matching target_shape.

    This is important because semantic segmentation models may return
    masks at a different resolution from the source image.
    """

    if mask is None:
        return None

    m = np.asarray(
        mask
    )

    if m.size == 0:
        return None

    if m.ndim > 2:
        m = np.squeeze(
            m
        )

    if m.ndim != 2:
        return None

    target_h, target_w = target_shape

    if m.shape != (
        target_h,
        target_w,
    ):

        m = cv2.resize(
            m.astype(
                np.float32,
                copy=False,
            ),
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    m = normalized_mask(
        m
    )

    if m is None:
        return None

    return np.asarray(
        m,
        dtype=bool,
    )


def _rgb_luminance(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Approximate display-referred luminance.

    This is intentionally different from linear luminance.

    Useful for:
        visual color analysis
        saturation context
        skin color analysis
        display-oriented diagnostics
    """

    bgr = (
        image_bgr.astype(
            np.float32
        )
        / 255.0
    )

    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    return (
        0.0722 * b
        + 0.7152 * g
        + 0.2126 * r
    ).astype(
        np.float32
    )


def _linearize_srgb(
    values: np.ndarray,
) -> np.ndarray:
    """
    Convert sRGB [0,1] to linear RGB.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = np.clip(
        values,
        0.0,
        1.0,
    )

    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power(
            (
                values + 0.055
            )
            / 1.055,
            2.4,
        ),
    ).astype(
        np.float32
    )


def _linear_luminance(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Calculate relative linear luminance.

    Input:
        BGR uint8.

    Output:
        float32 approximately 0..1.
    """

    bgr = (
        image_bgr.astype(
            np.float32
        )
        / 255.0
    )

    b = _linearize_srgb(
        bgr[:, :, 0]
    )

    g = _linearize_srgb(
        bgr[:, :, 1]
    )

    r = _linearize_srgb(
        bgr[:, :, 2]
    )

    return (
        0.0722 * b
        + 0.7152 * g
        + 0.2126 * r
    ).astype(
        np.float32
    )


# ============================================================================
# COLOR ANALYSIS
# ============================================================================

def analyze_color(
    image: np.ndarray,
    semantic: Mapping[
        str,
        Optional[np.ndarray],
    ],
) -> Dict[str, Any]:
    """
    Analyze image color characteristics.

    Important distinction:

        saturation != color quality

    A highly saturated image is not automatically bad.

    The analyzer therefore considers:

        saturation
        luminance
        exposure
        clipping
        skin coverage
        useful-exposure saturation
        colorfulness
        saturation distribution

    No image pixels are modified.
    """

    if image is None:
        raise ValueError(
            "image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be numpy.ndarray."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected BGR image with 3 dimensions."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected BGR image with 3 channels."
        )

    small = _resize_for_analysis(
        image,
        COLOR_ANALYSIS_MAX_DIM,
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(
            np.float32
        )
        / 255.0
    )

    value = (
        hsv[:, :, 2].astype(
            np.float32
        )
        / 255.0
    )

    luminance = _rgb_luminance(
        small
    )

    # ------------------------------------------------------------------------
    # Paired deterministic sampling
    #
    # This is important.
    #
    # The old implementation independently sampled saturation and luminance.
    # That means saturation[i] was not necessarily the same pixel as
    # luminance[i].
    #
    # Here one index set is used for all three arrays.
    # ------------------------------------------------------------------------

    flat_sat = saturation.reshape(
        -1
    )

    flat_lum = luminance.reshape(
        -1
    )

    flat_value = value.reshape(
        -1
    )

    indices = _deterministic_indices(
        flat_sat.size,
        COLOR_SAMPLE_PIXELS,
    )

    sat_values = flat_sat[
        indices
    ]

    lum_values = flat_lum[
        indices
    ]

    value_values = flat_value[
        indices
    ]

    # Remove invalid paired pixels.
    finite = (
        np.isfinite(
            sat_values
        )
        & np.isfinite(
            lum_values
        )
        & np.isfinite(
            value_values
        )
    )

    sat_values = sat_values[
        finite
    ]

    lum_values = lum_values[
        finite
    ]

    value_values = value_values[
        finite
    ]

    if sat_values.size == 0:

        sat_values = np.array(
            [0.0],
            dtype=np.float32,
        )

        lum_values = np.array(
            [0.0],
            dtype=np.float32,
        )

        value_values = np.array(
            [0.0],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------------
    # Saturation distribution
    # ------------------------------------------------------------------------

    p10 = _safe_percentile(
        sat_values,
        10,
    )

    p25 = _safe_percentile(
        sat_values,
        25,
    )

    p50 = _safe_percentile(
        sat_values,
        50,
    )

    p75 = _safe_percentile(
        sat_values,
        75,
    )

    p90 = _safe_percentile(
        sat_values,
        90,
    )

    mean = _safe_mean(
        sat_values
    )

    # ------------------------------------------------------------------------
    # Saturation coverage
    # ------------------------------------------------------------------------

    high_sat = _safe_fraction(
        sat_values >= 0.82
    )

    extreme_sat = _safe_fraction(
        sat_values >= 0.94
    )

    low_sat = _safe_fraction(
        sat_values <= 0.10
    )

    # ------------------------------------------------------------------------
    # Useful-exposure saturation
    #
    # Saturation in nearly black or clipped pixels should have less influence
    # on automatic color decisions.
    # ------------------------------------------------------------------------

    useful_exposure = (
        (lum_values >= 0.04)
        & (lum_values <= 0.96)
        & (value_values >= 0.035)
        & (value_values <= 0.98)
    )

    if np.any(
        useful_exposure
    ):

        useful_sat_values = sat_values[
            useful_exposure
        ]

        useful_high_sat = _safe_fraction(
            useful_sat_values >= 0.82
        )

        useful_extreme_sat = _safe_fraction(
            useful_sat_values >= 0.94
        )

        useful_median_sat = _safe_percentile(
            useful_sat_values,
            50,
        )

    else:

        useful_high_sat = 0.0
        useful_extreme_sat = 0.0
        useful_median_sat = 0.0

    # ------------------------------------------------------------------------
    # Skin analysis
    # ------------------------------------------------------------------------

    skin = None

    if semantic is not None:
        skin = semantic.get(
            "skin"
        )

    skin_sat = None
    skin_coverage = 0.0
    skin_high_sat = None

    if skin is not None:

        skin_mask = _prepare_analysis_mask(
            skin,
            saturation.shape,
        )

        if skin_mask is not None:

            skin_values = saturation[
                skin_mask
            ]

            if skin_values.size:

                skin_values = (
                    _deterministic_sample(
                        skin_values,
                        SKIN_SAMPLE_PIXELS,
                    )
                )

                skin_values = skin_values[
                    np.isfinite(
                        skin_values
                    )
                ]

                if skin_values.size:

                    skin_sat = _safe_percentile(
                        skin_values,
                        50,
                    )

                    skin_high_sat = _safe_fraction(
                        skin_values > 0.70
                    )

                    skin_coverage = float(
                        np.mean(
                            skin_mask
                        )
                    )

    # ------------------------------------------------------------------------
    # Colorfulness
    #
    # Mean saturation alone is insufficient.
    #
    # Give normally exposed pixels more importance and very dark/bright
    # pixels less influence.
    # ------------------------------------------------------------------------

    exposure_weight = np.clip(
        (
            luminance - 0.02
        )
        / 0.94,
        0.0,
        1.0,
    )

    # Soft weighting instead of hard exclusion.
    colorfulness = _safe_mean(
        saturation
        * (
            0.25
            + 0.75
            * exposure_weight
        )
    )

    # ------------------------------------------------------------------------
    # Global clipping / exposure fractions
    # ------------------------------------------------------------------------

    dark_fraction = _safe_fraction(
        value < 0.025
    )

    bright_fraction = _safe_fraction(
        value > 0.985
    )

    # ------------------------------------------------------------------------
    # Auto vibrance
    #
    # Conservative by design.
    #
    # Vibrance should normally do more work than saturation, but should
    # become restrained when the image is already highly colorful.
    # ------------------------------------------------------------------------

    vibrance = 0.0

    if useful_median_sat < 0.12:

        vibrance = clamp(
            (
                0.16
                - useful_median_sat
            )
            * 30.0,
            0.0,
            7.0,
        )

    elif useful_median_sat < 0.19:

        vibrance = clamp(
            (
                0.19
                - useful_median_sat
            )
            * 17.0,
            0.0,
            4.0,
        )

    elif useful_high_sat > 0.15:

        vibrance = -clamp(
            (
                useful_high_sat
                - 0.15
            )
            * 18.0,
            0.0,
            6.0,
        )

    # Extreme saturation gets additional restraint.
    if useful_extreme_sat > 0.035:

        vibrance -= clamp(
            useful_extreme_sat
            * 18.0,
            0.0,
            4.0,
        )

    # ------------------------------------------------------------------------
    # Skin protection
    # ------------------------------------------------------------------------

    if skin_coverage > 0.08:

        vibrance *= 0.50

    elif skin_coverage > 0.03:

        vibrance *= 0.68

    # If skin itself is already highly saturated, further global vibrance
    # should be even more restrained.
    if (
        skin_high_sat is not None
        and skin_high_sat > 0.12
    ):

        vibrance *= 0.60

    # ------------------------------------------------------------------------
    # Auto saturation
    #
    # Deliberately much smaller than vibrance.
    # ------------------------------------------------------------------------

    saturation_adjustment = 0.0

    if (
        useful_median_sat > 0.70
        and useful_high_sat > 0.12
    ):

        saturation_adjustment = -3.0

    elif (
        useful_median_sat > 0.80
        or useful_extreme_sat > 0.08
    ):

        saturation_adjustment = -5.0

    elif (
        useful_median_sat < 0.065
        and useful_high_sat < 0.03
    ):

        saturation_adjustment = 1.0

    # Skin protection.
    if skin_coverage > 0.08:

        saturation_adjustment *= 0.40

    elif skin_coverage > 0.03:

        saturation_adjustment *= 0.60

    # ------------------------------------------------------------------------
    # Color state
    # ------------------------------------------------------------------------

    if (
        useful_extreme_sat > 0.05
        or useful_high_sat > 0.20
    ):

        color_state = "highly_saturated"

    elif useful_median_sat < 0.10:

        color_state = "muted"

    elif useful_median_sat < 0.18:

        color_state = "moderately_muted"

    elif useful_median_sat > 0.65:

        color_state = "high_color"

    else:

        color_state = "normal"

    return {
        # --------------------------------------------------------------------
        # Backwards-compatible fields
        # --------------------------------------------------------------------

        "mean_saturation": round(
            mean,
            5,
        ),

        "median_saturation": round(
            p50,
            5,
        ),

        "p10_saturation": round(
            p10,
            5,
        ),

        "p90_saturation": round(
            p90,
            5,
        ),

        "high_saturation_fraction": round(
            high_sat,
            5,
        ),

        "skin_median_saturation": (
            round(
                skin_sat,
                5,
            )
            if skin_sat is not None
            else None
        ),

        "vibrance": round(
            clamp(
                vibrance,
                -15.0,
                12.0,
            ),
            3,
        ),

        "saturation": round(
            clamp(
                saturation_adjustment,
                -8.0,
                4.0,
            ),
            3,
        ),

        # --------------------------------------------------------------------
        # Extended fields
        # --------------------------------------------------------------------

        "p25_saturation": round(
            p25,
            5,
        ),

        "p75_saturation": round(
            p75,
            5,
        ),

        "low_saturation_fraction": round(
            low_sat,
            5,
        ),

        "extreme_saturation_fraction": round(
            extreme_sat,
            5,
        ),

        "useful_exposure_high_saturation_fraction": round(
            useful_high_sat,
            5,
        ),

        "useful_exposure_extreme_saturation_fraction": round(
            useful_extreme_sat,
            5,
        ),

        "useful_exposure_median_saturation": round(
            useful_median_sat,
            5,
        ),

        "colorfulness": round(
            colorfulness,
            5,
        ),

        "skin_coverage": round(
            skin_coverage,
            5,
        ),

        "skin_high_saturation_fraction": (
            round(
                skin_high_sat,
                5,
            )
            if skin_high_sat is not None
            else None
        ),

        "dark_fraction": round(
            dark_fraction,
            5,
        ),

        "bright_fraction": round(
            bright_fraction,
            5,
        ),

        "color_state": color_state,
    }


# ============================================================================
# DETAIL ANALYSIS
# ============================================================================

def analyze_detail(
    image: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze sharpness, edge structure and probable noise.

    Laplacian variance alone is unreliable because:

        noise              -> high Laplacian
        JPEG artifacts     -> high Laplacian
        real detail        -> high Laplacian

    Therefore this analyzer combines:

        denoised Laplacian variance
        raw Laplacian variance
        gradient statistics
        adaptive edge density
        strong edge density
        high-frequency residual
        noise/detail relationship

    The result is deterministic and resolution-aware.
    """

    if image is None:
        raise ValueError(
            "image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be numpy.ndarray."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected BGR image."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected 3-channel BGR image."
        )

    small = _resize_for_analysis(
        image,
        DETAIL_MAX_DIM,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    gray_float = (
        gray.astype(
            np.float32
        )
        / 255.0
    )

    # ------------------------------------------------------------------------
    # Slight denoising reference
    # ------------------------------------------------------------------------

    denoised = cv2.GaussianBlur(
        gray_float,
        (
            3,
            3,
        ),
        0.65,
    )

    # ------------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------------

    raw_lap = cv2.Laplacian(
        gray_float,
        cv2.CV_32F,
        ksize=3,
    )

    denoised_lap = cv2.Laplacian(
        denoised,
        cv2.CV_32F,
        ksize=3,
    )

    raw_sharpness = float(
        np.var(
            raw_lap
        )
    )

    sharpness = float(
        np.var(
            denoised_lap
        )
    )

    # ------------------------------------------------------------------------
    # Sobel gradient
    # ------------------------------------------------------------------------

    gx = cv2.Sobel(
        denoised,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        denoised,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = np.sqrt(
        gx * gx
        + gy * gy
    )

    gradient_values = _deterministic_sample(
        gradient,
        DETAIL_SAMPLE_PIXELS,
    )

    gradient_values = gradient_values[
        np.isfinite(
            gradient_values
        )
    ]

    gradient_mean = _safe_mean(
        gradient_values
    )

    gradient_median = _safe_percentile(
        gradient_values,
        50,
    )

    gradient_p90 = _safe_percentile(
        gradient_values,
        90,
    )

    gradient_p95 = _safe_percentile(
        gradient_values,
        95,
    )

    # ------------------------------------------------------------------------
    # Adaptive edge threshold
    #
    # Using a percentile makes the measurement more camera-independent than
    # a fixed threshold alone.
    # ------------------------------------------------------------------------

    edge_threshold = max(
        0.035,
        gradient_p90 * 0.32,
    )

    strong_edge_threshold = max(
        0.070,
        gradient_p95 * 0.45,
    )

    edge_fraction = _safe_fraction(
        gradient > edge_threshold
    )

    strong_edge_fraction = _safe_fraction(
        gradient > strong_edge_threshold
    )

    # ------------------------------------------------------------------------
    # High-frequency residual
    # ------------------------------------------------------------------------

    residual = (
        gray_float
        - denoised
    )

    residual_abs = np.abs(
        residual
    )

    residual_values = _deterministic_sample(
        residual_abs,
        DETAIL_SAMPLE_PIXELS,
    )

    residual_values = residual_values[
        np.isfinite(
            residual_values
        )
    ]

    high_frequency_noise = _safe_percentile(
        residual_values,
        90,
    )

    residual_median = _safe_percentile(
        residual_values,
        50,
    )

    residual_p95 = _safe_percentile(
        residual_values,
        95,
    )

    # ------------------------------------------------------------------------
    # Noise/detail relationship
    #
    # raw_sharpness > denoised_sharpness means high-frequency information
    # contributes significantly to the apparent sharpness.
    # ------------------------------------------------------------------------

    if sharpness > EPS:

        raw_to_denoised_ratio = (
            raw_sharpness
            / (
                sharpness
                + EPS
            )
        )

    else:

        raw_to_denoised_ratio = 1.0

    raw_to_denoised_ratio = clamp(
        raw_to_denoised_ratio,
        1.0,
        20.0,
    )

    # Noise ratio combines the high-frequency residual with the amount
    # of genuine edge structure.
    #
    # More edge structure should make us less suspicious of high-frequency
    # energy.
    edge_protection = clamp(
        edge_fraction
        / 0.12,
        0.0,
        1.0,
    )

    noise_ratio = clamp(
        (
            (
                raw_to_denoised_ratio
                - 1.0
            )
            / 3.0
        )
        * 0.65
        +
        high_frequency_noise
        * 2.0
        * (
            1.0
            - 0.35 * edge_protection
        ),
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------------

    h, w = gray.shape

    pixels = (
        h * w
    )

    # ------------------------------------------------------------------------
    # Detail score
    #
    # Absolute sharpness is still useful, but only after compression into
    # a stable logarithmic range.
    # ------------------------------------------------------------------------

    sharpness_component = clamp(
        math.log1p(
            max(
                sharpness,
                0.0,
            )
        )
        / math.log1p(
            0.01
        ),
        0.0,
        1.0,
    )

    edge_component = clamp(
        edge_fraction
        / 0.18,
        0.0,
        1.0,
    )

    strong_edge_component = clamp(
        strong_edge_fraction
        / 0.08,
        0.0,
        1.0,
    )

    detail_score = clamp(
        (
            sharpness_component
            * 0.35
            +
            edge_component
            * 0.40
            +
            strong_edge_component
            * 0.25
        )
        * (
            1.0
            - 0.25 * noise_ratio
        ),
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Detail quality classification
    #
    # Use several signals rather than one Laplacian threshold.
    # ------------------------------------------------------------------------

    if (
        detail_score < 0.18
        and edge_fraction < 0.025
    ):

        detail_quality = "very_soft"

    elif (
        detail_score < 0.34
        and edge_fraction < 0.055
    ):

        detail_quality = "soft"

    elif (
        detail_score < 0.62
    ):

        detail_quality = "normal"

    elif (
        detail_score < 0.82
    ):

        detail_quality = "sharp"

    else:

        detail_quality = "very_sharp"

    # ------------------------------------------------------------------------
    # Noise classification
    # ------------------------------------------------------------------------

    if (
        noise_ratio > 0.68
        and raw_to_denoised_ratio > 2.0
    ):

        noise_quality = "high_noise"

    elif (
        noise_ratio > 0.42
        and raw_to_denoised_ratio > 1.45
    ):

        noise_quality = "moderate_noise"

    elif noise_ratio > 0.22:

        noise_quality = "mild_noise"

    else:

        noise_quality = "clean"

    # ------------------------------------------------------------------------
    # Detail confidence
    #
    # Confidence answers:
    #
    #   "How trustworthy is this detail measurement?"
    #
    # rather than:
    #
    #   "How sharp is the image?"
    # ------------------------------------------------------------------------

    detail_confidence = clamp(
        (
            0.35
            + 0.30
            * min(
                1.0,
                gradient_p90
                / 0.25,
            )
            + 0.20
            * min(
                1.0,
                edge_fraction
                / 0.15,
            )
            + 0.15
            * min(
                1.0,
                pixels
                / 1_000_000.0,
            )
        )
        * (
            1.0
            - 0.30 * noise_ratio
        ),
        0.0,
        1.0,
    )

    return {
        # --------------------------------------------------------------------
        # Backwards-compatible fields
        # --------------------------------------------------------------------

        "sharpness": round(
            sharpness,
            7,
        ),

        "edge_fraction": round(
            edge_fraction,
            5,
        ),

        "quality": detail_quality,

        # --------------------------------------------------------------------
        # Extended fields
        # --------------------------------------------------------------------

        "raw_sharpness": round(
            raw_sharpness,
            7,
        ),

        "gradient_mean": round(
            gradient_mean,
            6,
        ),

        "gradient_median": round(
            gradient_median,
            6,
        ),

        "gradient_p90": round(
            gradient_p90,
            6,
        ),

        "gradient_p95": round(
            gradient_p95,
            6,
        ),

        "strong_edge_fraction": round(
            strong_edge_fraction,
            5,
        ),

        "adaptive_edge_threshold": round(
            edge_threshold,
            6,
        ),

        "strong_edge_threshold": round(
            strong_edge_threshold,
            6,
        ),

        "high_frequency_noise": round(
            high_frequency_noise,
            6,
        ),

        "high_frequency_median": round(
            residual_median,
            6,
        ),

        "high_frequency_p95": round(
            residual_p95,
            6,
        ),

        "raw_to_denoised_sharpness_ratio": round(
            raw_to_denoised_ratio,
            5,
        ),

        "noise_ratio": round(
            noise_ratio,
            5,
        ),

        "noise_quality": noise_quality,

        "detail_score": round(
            detail_score,
            4,
        ),

        "detail_confidence": round(
            detail_confidence,
            4,
        ),

        "analysis_width": int(
            w
        ),

        "analysis_height": int(
            h
        ),

        "analysis_pixels": int(
            pixels
        ),
    }


# ============================================================================
# HISTOGRAM
# ============================================================================

def calculate_histogram(
    values: np.ndarray,
    bins: int = 64,
) -> list[float]:
    """
    Generate a normalized histogram.

    Supported input:

        uint8:
            0..255

        float:
            expected 0..1

    Output:

        list of normalized bin frequencies.

    The output always contains exactly `bins` entries.
    """

    bins = int(
        bins
    )

    if bins <= 0:
        return []

    if values is None:
        return [
            0.0
        ] * bins

    arr = np.asarray(
        values
    )

    if arr.size == 0:
        return [
            0.0
        ] * bins

    arr = arr.reshape(
        -1
    )

    if np.issubdtype(
        arr.dtype,
        np.integer,
    ):

        arr = (
            arr.astype(
                np.float32
            )
            / 255.0
        )

    else:

        arr = arr.astype(
            np.float32,
            copy=False,
        )

    arr = arr[
        np.isfinite(
            arr
        )
    ]

    if arr.size == 0:
        return [
            0.0
        ] * bins

    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    arr = _deterministic_sample(
        arr,
        HISTOGRAM_SAMPLE_PIXELS,
    )

    hist, _ = np.histogram(
        arr,
        bins=bins,
        range=(
            0.0,
            1.0,
        ),
    )

    total = float(
        np.sum(
            hist
        )
    )

    if total <= 0.0:
        return [
            0.0
        ] * bins

    normalized = (
        hist.astype(
            np.float64
        )
        / total
    )

    return [
        round(
            float(
                value
            ),
            7,
        )
        for value in normalized
    ]


# ============================================================================
# GRID ANALYSIS
# ============================================================================

def analyze_grid(
    y_linear: np.ndarray,
) -> list[
    Dict[str, Any]
]:
    """
    Analyze luminance distribution spatially.

    3x3 grid.

    Each cell contains:

        mean
        median
        p10
        p90
        contrast
        shadow fraction
        highlight fraction
        usable fraction

    This allows Auto Tone to distinguish:

        globally dark
        subject/background imbalance
        bright sky
        dark foreground
        uneven illumination
        localized clipping
    """

    if y_linear is None:
        return []

    y = np.asarray(
        y_linear,
        dtype=np.float32,
    )

    if y.ndim != 2:
        raise ValueError(
            "y_linear must be a 2D array."
        )

    if y.size == 0:
        return []

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    y = np.clip(
        y,
        0.0,
        1.0,
    )

    h, w = y.shape

    rows = 3
    cols = 3

    result: list[
        Dict[str, Any]
    ] = []

    for row in range(
        rows
    ):

        y0 = int(
            row
            * h
            / rows
        )

        y1 = int(
            (row + 1)
            * h
            / rows
        )

        for col in range(
            cols
        ):

            x0 = int(
                col
                * w
                / cols
            )

            x1 = int(
                (col + 1)
                * w
                / cols
            )

            region = y[
                y0:y1,
                x0:x1,
            ]

            if region.size == 0:

                result.append(
                    {
                        "row": row,
                        "column": col,
                        "mean": 0.0,
                        "median": 0.0,
                        "p10": 0.0,
                        "p90": 0.0,
                        "contrast": 0.0,
                        "shadow_fraction": 0.0,
                        "highlight_fraction": 0.0,
                        "usable_fraction": 0.0,
                    }
                )

                continue

            values = _deterministic_sample(
                region,
                GRID_SAMPLE_PIXELS,
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            if values.size == 0:

                result.append(
                    {
                        "row": row,
                        "column": col,
                        "mean": 0.0,
                        "median": 0.0,
                        "p10": 0.0,
                        "p90": 0.0,
                        "contrast": 0.0,
                        "shadow_fraction": 0.0,
                        "highlight_fraction": 0.0,
                        "usable_fraction": 0.0,
                    }
                )

                continue

            mean = _safe_mean(
                values
            )

            median = _safe_percentile(
                values,
                50,
            )

            p10 = _safe_percentile(
                values,
                10,
            )

            p90 = _safe_percentile(
                values,
                90,
            )

            contrast = max(
                0.0,
                p90 - p10,
            )

            shadow_fraction = _safe_fraction(
                values < 0.05
            )

            highlight_fraction = _safe_fraction(
                values > 0.95
            )

            usable_fraction = _safe_fraction(
                (
                    values >= 0.02
                )
                & (
                    values <= 0.98
                )
            )

            result.append(
                {
                    "row": row,
                    "column": col,

                    "mean": round(
                        mean,
                        5,
                    ),

                    "median": round(
                        median,
                        5,
                    ),

                    "p10": round(
                        p10,
                        5,
                    ),

                    "p90": round(
                        p90,
                        5,
                    ),

                    "contrast": round(
                        contrast,
                        5,
                    ),

                    "shadow_fraction": round(
                        shadow_fraction,
                        5,
                    ),

                    "highlight_fraction": round(
                        highlight_fraction,
                        5,
                    ),

                    "usable_fraction": round(
                        usable_fraction,
                        5,
                    ),
                }
            )

    return result