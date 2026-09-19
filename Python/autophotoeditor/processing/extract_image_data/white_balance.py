"""
AutoPhotoEditor - Image Analysis V4
===================================

White Balance Analysis
----------------------

High-quality, conservative white-balance analysis for automatic
photo enhancement.

Design goals
------------

1. Neutral-reference-first WB estimation.
2. Floating-point LAB optimization.
3. Robust gray/neutral patch detection.
4. Semantic-mask awareness.
5. Multiple independent WB estimators.
6. Estimator agreement validation.
7. Strong identity/no-op safety.
8. Confidence-aware correction strength.
9. Protection against skin/vegetation/sky contamination.
10. Deterministic processing.
11. Fast enough for normal desktop image analysis.

Input
-----
BGR uint8 image.

Output
------
JSON-compatible dictionary.

Important
---------
This module does NOT modify the image.

It produces a WB correction plan for the enhancer.

Expected RGB gains:

    [red_gain, 1.0, blue_gain]

Green is used as the luminance/WB anchor.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .shared import *


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("image_analysis_v4.wb")


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8

# Maximum number of pixels actually used by each WB estimator.
MAX_WB_PIXELS = 60_000

# Maximum pixels used by LAB optimization.
MAX_NEUTRAL_OPT_PIXELS = 18_000

# Learning based WB can be relatively expensive.
DEFAULT_USE_LEARNING_WB = True

# Physical-ish safety limits.
#
# These are deliberately conservative.
MIN_WB_GAIN = 0.72
MAX_WB_GAIN = 1.40

# Stronger final limits for automatic correction.
FINAL_MIN_WB_GAIN = 0.80
FINAL_MAX_WB_GAIN = 1.25

# Maximum logarithmic correction from identity.
MAX_LOG_CORRECTION = 0.24

# Small corrections below this are effectively invisible.
NOOP_LOG_THRESHOLD = 0.012

# Neutral detection.
NEUTRAL_MIN_L = 18.0
NEUTRAL_MAX_L = 92.0

# OpenCV HSV saturation is 0..255.
MAX_NEUTRAL_SATURATION = 105.0

# LAB chroma threshold.
BASE_NEUTRAL_CHROMA = 19.0

# Strong neutral threshold.
STRONG_NEUTRAL_CHROMA = 11.5

# Minimum spatial density for a useful patch.
MIN_LOCAL_NEUTRAL_DENSITY = 0.32

# Minimum useful neutral pixels.
MIN_NEUTRAL_PIXELS = 40

# Good neutral region.
GOOD_NEUTRAL_PIXELS = 1200

# Useful neutral coverage.
GOOD_NEUTRAL_COVERAGE = 0.018

# Minimum improvement required before trusting a correction.
MIN_USEFUL_IMPROVEMENT = 0.012

# Maximum correction allowed when neutral confidence is poor.
LOW_CONFIDENCE_LOG_LIMIT = 0.075

# ============================================================
# BASIC NUMERICAL HELPERS
# ============================================================


def _safe_array(
    values: Sequence[float],
) -> np.ndarray:
    """
    Convert gains to a safe float32 array.
    """
    arr = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    if arr.size != 3:
        return np.ones(
            3,
            dtype=np.float32,
        )

    if not np.all(
        np.isfinite(arr)
    ):
        return np.ones(
            3,
            dtype=np.float32,
        )

    return arr


def normalize_rgb_gains(
    gains: Sequence[float],
) -> np.ndarray:
    """
    Normalize WB gains with green fixed at 1.0.

    WB correction must not globally normalize all channels,
    because doing so can unintentionally change exposure.

    Returns
    -------
    np.ndarray
        [R_gain, 1.0, B_gain]
    """

    g = _safe_array(
        gains
    )

    green = max(
        float(g[1]),
        EPS,
    )

    g = g / green

    g[1] = 1.0

    g[0] = clamp(
        float(g[0]),
        MIN_WB_GAIN,
        MAX_WB_GAIN,
    )

    g[2] = clamp(
        float(g[2]),
        MIN_WB_GAIN,
        MAX_WB_GAIN,
    )

    g[1] = 1.0

    return g.astype(
        np.float32
    )


def _finalize_wb_gains(
    gains: Sequence[float],
    confidence: float,
) -> np.ndarray:
    """
    Apply conservative final safety limits.

    Low-confidence WB corrections are strongly pulled toward identity.
    """

    g = normalize_rgb_gains(
        gains
    )

    confidence = clamp(
        float(confidence),
        0.0,
        1.0,
    )

    log_g = np.log(
        np.maximum(
            g,
            EPS,
        )
    )

    # Pull the correction toward identity according to confidence.
    #
    # Never completely remove a high-confidence correction.
    strength = (
        0.20
        + 0.80 * confidence
    )

    log_g *= strength

    # Hard safety limit.
    log_g[0] = clamp(
        float(log_g[0]),
        -MAX_LOG_CORRECTION,
        MAX_LOG_CORRECTION,
    )

    log_g[2] = clamp(
        float(log_g[2]),
        -MAX_LOG_CORRECTION,
        MAX_LOG_CORRECTION,
    )

    result = np.exp(
        log_g
    ).astype(
        np.float32
    )

    result[1] = 1.0

    result[0] = clamp(
        float(result[0]),
        FINAL_MIN_WB_GAIN,
        FINAL_MAX_WB_GAIN,
    )

    result[2] = clamp(
        float(result[2]),
        FINAL_MIN_WB_GAIN,
        FINAL_MAX_WB_GAIN,
    )

    return result.astype(
        np.float32
    )


def _gain_distance_from_identity(
    gains: Sequence[float],
) -> float:
    """
    Log-domain distance from identity WB.
    """

    g = normalize_rgb_gains(
        gains
    )

    return float(
        math.sqrt(
            safe_log(float(g[0])) ** 2
            + safe_log(float(g[2])) ** 2
        )
    )


# ============================================================
# RGB / LAB CONVERSION
# ============================================================


def _rgb_to_lab_float(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert RGB float [0,1] -> CIE LAB float.

    This is important.

    The previous implementation optimized using 8-bit LAB.
    That introduces quantization into the objective function.

    OpenCV's float LAB representation gives approximately:

        L* = 0..100
        a* ~= -128..127
        b* ~= -128..127
    """

    rgb = np.asarray(
        rgb,
        dtype=np.float32,
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB,
    )


# ============================================================
# WB GAIN -> CAST METRICS
# ============================================================


def gains_to_temperature_tint(
    gains_rgb: Sequence[float],
) -> Tuple[float, float]:
    """
    Convert relative RGB gains into a perceptual cast metric.

    IMPORTANT:
    This is NOT a physical Kelvin conversion.

    It is intentionally reported as:

        temperature_cast
        tint_cast

    rather than pretending that the values represent real
    correlated color temperature.
    """

    g = normalize_rgb_gains(
        gains_rgb
    )

    r = max(
        float(g[0]),
        EPS,
    )

    b = max(
        float(g[2]),
        EPS,
    )

    # Warm/cool component.
    warmth = safe_log(
        r / b
    )

    temperature = clamp(
        warmth * 42.0,
        -100.0,
        100.0,
    )

    # Magenta/green component.
    #
    # Green is fixed at 1.0, so this measures deviation of
    # red+blue relative to green.
    tint_value = (
        safe_log(r)
        + safe_log(b)
    ) * 22.0

    tint = clamp(
        tint_value,
        -100.0,
        100.0,
    )

    return (
        float(temperature),
        float(tint),
    )


# ============================================================
# RGB GAIN APPLICATION
# ============================================================


def apply_rgb_gains(
    rgb: np.ndarray,
    gains: Sequence[float],
) -> np.ndarray:
    """
    Apply RGB gains without changing array dimensions.
    """

    g = normalize_rgb_gains(
        gains
    ).reshape(
        1,
        1,
        3,
    )

    return np.clip(
        rgb * g,
        0.0,
        1.0,
    )


# ============================================================
# ROBUST SAMPLING
# ============================================================


def _sample_pixels(
    pixels: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    """
    Deterministic pixel sampling.

    Avoids depending on whether shared.sample_array interprets
    its argument as elements or rows.
    """

    if pixels.size == 0:
        return pixels

    pixels = np.asarray(
        pixels
    )

    if pixels.ndim != 2:
        pixels = pixels.reshape(
            -1,
            pixels.shape[-1],
        )

    count = pixels.shape[0]

    if count <= max_pixels:
        return pixels

    # Deterministic evenly distributed sampling.
    indices = np.linspace(
        0,
        count - 1,
        max_pixels,
        dtype=np.int64,
    )

    return pixels[
        indices
    ]


# ============================================================
# GRAY WORLD
# ============================================================


def gray_world_gains(
    rgb: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Robust gray-world estimator.

    Uses median rather than mean to reduce the effect of:
    - highlights
    - saturated colors
    - bright objects
    """

    if valid is None:
        pixels = rgb.reshape(
            -1,
            3,
        )
    else:
        mask = normalized_mask(
            valid
        )

        if mask is None:
            return np.ones(
                3,
                dtype=np.float32,
            )

        pixels = rgb[
            mask
        ]

    if pixels.size == 0:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = _sample_pixels(
        pixels,
        MAX_WB_PIXELS,
    )

    pixels = pixels[
        np.all(
            np.isfinite(pixels),
            axis=1,
        )
    ]

    if pixels.shape[0] < 20:
        return np.ones(
            3,
            dtype=np.float32,
        )

    # Reject extreme pixels.
    pixels = pixels[
        (pixels.min(axis=1) > 0.025)
        & (pixels.max(axis=1) < 0.975)
    ]

    if pixels.shape[0] < 20:
        return np.ones(
            3,
            dtype=np.float32,
        )

    medians = np.median(
        pixels,
        axis=0,
    )

    medians = np.maximum(
        medians,
        EPS,
    )

    target = float(
        np.exp(
            np.mean(
                np.log(medians)
            )
        )
    )

    gains = (
        target
        / medians
    )

    return normalize_rgb_gains(
        gains
    )


# ============================================================
# SHADES OF GRAY
# ============================================================


def shades_of_gray_gains(
    rgb: np.ndarray,
    p: float = 6.0,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Shades-of-gray illuminant estimator.
    """

    if valid is None:
        pixels = rgb.reshape(
            -1,
            3,
        )
    else:
        mask = normalized_mask(
            valid
        )

        if mask is None:
            return np.ones(
                3,
                dtype=np.float32,
            )

        pixels = rgb[
            mask
        ]

    if pixels.size == 0:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = _sample_pixels(
        pixels,
        MAX_WB_PIXELS,
    )

    pixels = pixels[
        np.all(
            np.isfinite(pixels),
            axis=1,
        )
    ]

    pixels = pixels[
        (pixels.min(axis=1) > 0.025)
        & (pixels.max(axis=1) < 0.975)
    ]

    if pixels.shape[0] < 20:
        return np.ones(
            3,
            dtype=np.float32,
        )

    power = np.mean(
        np.power(
            np.maximum(
                pixels,
                EPS,
            ),
            p,
        ),
        axis=0,
    )

    norms = np.power(
        np.maximum(
            power,
            EPS,
        ),
        1.0 / p,
    )

    target = float(
        np.exp(
            np.mean(
                np.log(
                    np.maximum(
                        norms,
                        EPS,
                    )
                )
            )
        )
    )

    gains = (
        target
        / norms
    )

    return normalize_rgb_gains(
        gains
    )


# ============================================================
# NEUTRAL PATCH INFORMATION
# ============================================================


@dataclass
class NeutralPatchStats:

    mask: np.ndarray

    count: int

    coverage: float

    neutrality_before: float

    luminance_median: float

    luminance_std: float

    spatial_quality: float

    confidence: float


# ============================================================
# SEMANTIC EXCLUSION
# ============================================================


def _build_neutral_exclusion(
    semantic: Mapping[str, Optional[np.ndarray]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Build conservative semantic exclusions.

    Skin is particularly important because skin frequently has
    low/moderate chroma but must never become the WB reference.
    """

    masks = []

    for name in (
        "skin",
        "vegetation",
        "sky",
        "hair",
        "lips",
        "eyes",
    ):
        mask = semantic.get(
            name
        )

        if mask is not None:
            normalized = normalized_mask(
                mask
            )

            if normalized is not None:
                masks.append(
                    normalized
                )

    if not masks:
        return np.zeros(
            shape,
            dtype=bool,
        )

    result = np.zeros(
        shape,
        dtype=bool,
    )

    for mask in masks:

        if mask.shape != shape:
            continue

        result |= mask

    return result


# ============================================================
# NEUTRAL PATCH DETECTION
# ============================================================


def find_neutral_patches(
    image: np.ndarray,
    semantic: Mapping[str, Optional[np.ndarray]],
    subject: Optional[np.ndarray],
) -> NeutralPatchStats:
    """
    Detect likely neutral gray/white surfaces.

    Important improvements over the previous version:

    - floating-point LAB
    - adaptive chroma threshold
    - local spatial consistency
    - semantic exclusion
    - channel-ratio screening
    - morphology
    - rejection of tiny components
    - confidence scoring
    """

    h, w = image.shape[:2]

    rgb = bgr_to_rgb_float(
        image
    )

    lab = _rgb_to_lab_float(
        rgb
    )

    l = lab[
        :, :, 0
    ]

    a = lab[
        :, :, 1
    ]

    b = lab[
        :, :, 2
    ]

    chroma = np.sqrt(
        a * a
        + b * b
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[
        :, :, 1
    ].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Semantic exclusions
    # --------------------------------------------------------

    excluded = _build_neutral_exclusion(
        semantic,
        (h, w),
    )

    # --------------------------------------------------------
    # Exposure validity
    # --------------------------------------------------------

    min_channel = image.min(
        axis=2
    )

    max_channel = image.max(
        axis=2
    )

    usable_exposure = (
        (min_channel >= 8)
        & (max_channel <= 248)
    )

    # --------------------------------------------------------
    # RGB channel relationship
    # --------------------------------------------------------
    #
    # A neutral object should not have a huge channel ratio.
    #
    # This catches some pixels which happen to have acceptable
    # LAB chroma but are actually strongly colored.
    # --------------------------------------------------------

    rgb_min = np.maximum(
        rgb.min(axis=2),
        EPS,
    )

    rgb_max = rgb.max(
        axis=2
    )

    channel_ratio = (
        rgb_max
        / rgb_min
    )

    ratio_candidate = (
        channel_ratio <= 1.55
    )

    # --------------------------------------------------------
    # Base candidate
    # --------------------------------------------------------

    candidate = (
        (l >= NEUTRAL_MIN_L)
        & (l <= NEUTRAL_MAX_L)
        & (
            chroma
            <= BASE_NEUTRAL_CHROMA
        )
        & (
            saturation
            <= MAX_NEUTRAL_SATURATION
        )
        & usable_exposure
        & ratio_candidate
        & (~excluded)
    )

    # --------------------------------------------------------
    # Strong neutral
    # --------------------------------------------------------

    strong = (
        (l >= 23.0)
        & (l <= 88.0)
        & (
            chroma
            <= STRONG_NEUTRAL_CHROMA
        )
        & (
            saturation
            <= 70.0
        )
        & usable_exposure
        & (~excluded)
    )

    # --------------------------------------------------------
    # Spatial density
    # --------------------------------------------------------

    candidate_u8 = candidate.astype(
        np.uint8
    )

    local_density = cv2.boxFilter(
        candidate_u8,
        cv2.CV_32F,
        (11, 11),
        normalize=True,
    )

    patch_mask = (
        candidate
        & (
            local_density
            >= MIN_LOCAL_NEUTRAL_DENSITY
        )
    )

    # Strong neutrals are retained even if texture reduces density.
    patch_mask |= strong

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel3 = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    kernel5 = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    patch_mask = cv2.morphologyEx(
        patch_mask.astype(
            np.uint8
        ),
        cv2.MORPH_OPEN,
        kernel3,
    )

    patch_mask = cv2.morphologyEx(
        patch_mask,
        cv2.MORPH_CLOSE,
        kernel5,
    )

    patch_mask = patch_mask.astype(
        bool
    )

    # --------------------------------------------------------
    # Remove tiny connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        patch_mask.astype(
            np.uint8
        ),
        connectivity=8,
    )

    cleaned = np.zeros(
        (h, w),
        dtype=bool,
    )

    minimum_component = max(
        25,
        int(
            h * w * 0.00002
        ),
    )

    for label in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= minimum_component:
            cleaned[
                labels == label
            ] = True

    patch_mask = cleaned

    count = int(
        np.count_nonzero(
            patch_mask
        )
    )

    if count == 0:

        return NeutralPatchStats(
            mask=patch_mask,
            count=0,
            coverage=0.0,
            neutrality_before=100.0,
            luminance_median=0.0,
            luminance_std=0.0,
            spatial_quality=0.0,
            confidence=0.0,
        )

    selected_chroma = chroma[
        patch_mask
    ]

    selected_l = l[
        patch_mask
    ]

    selected_density = local_density[
        patch_mask
    ]

    coverage = float(
        count
        / max(
            h * w,
            1,
        )
    )

    neutrality = robust_median(
        selected_chroma,
        100.0,
    )

    luminance_median = robust_median(
        selected_l,
        0.0,
    )

    luminance_std = robust_std(
        selected_l,
        0.0,
    )

    spatial_quality = clamp(
        float(
            np.mean(
                selected_density
            )
        ),
        0.0,
        1.0,
    )

    # More pixels help, but with diminishing returns.
    count_confidence = clamp(
        math.log1p(count)
        / math.log1p(
            GOOD_NEUTRAL_PIXELS
        ),
        0.0,
        1.0,
    )

    coverage_confidence = clamp(
        coverage
        / GOOD_NEUTRAL_COVERAGE,
        0.0,
        1.0,
    )

    neutrality_confidence = clamp(
        1.0
        - neutrality / 24.0,
        0.0,
        1.0,
    )

    luminance_confidence = clamp(
        1.0
        - max(
            0.0,
            luminance_std - 18.0,
        ) / 35.0,
        0.0,
        1.0,
    )

    confidence = (
        0.25 * count_confidence
        + 0.20 * coverage_confidence
        + 0.25 * neutrality_confidence
        + 0.20 * spatial_quality
        + 0.10 * luminance_confidence
    )

    confidence = clamp(
        confidence,
        0.0,
        0.99,
    )

    return NeutralPatchStats(
        mask=patch_mask,
        count=count,
        coverage=coverage,
        neutrality_before=float(
            neutrality
        ),
        luminance_median=float(
            luminance_median
        ),
        luminance_std=float(
            luminance_std
        ),
        spatial_quality=float(
            spatial_quality
        ),
        confidence=float(
            confidence
        ),
    )


# ============================================================
# NEUTRAL PATCH GAINS
# ============================================================


def neutral_patch_gains(
    rgb: np.ndarray,
    neutral_mask: np.ndarray,
) -> np.ndarray:
    """
    Estimate WB from a neutral region.

    Uses robust log-domain channel statistics.
    """

    mask = normalized_mask(
        neutral_mask
    )

    if mask is None:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = rgb[
        mask
    ]

    if pixels.size == 0:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = _sample_pixels(
        pixels,
        MAX_WB_PIXELS,
    )

    valid = (
        np.all(
            np.isfinite(
                pixels
            ),
            axis=1,
        )
        & (
            pixels.min(axis=1)
            > 0.025
        )
        & (
            pixels.max(axis=1)
            < 0.985
        )
    )

    pixels = pixels[
        valid
    ]

    if pixels.shape[0] < MIN_NEUTRAL_PIXELS:
        return np.ones(
            3,
            dtype=np.float32,
        )

    # Work in log space.
    #
    # This makes the estimator more stable when illumination
    # changes multiplicatively.
    log_pixels = np.log(
        np.maximum(
            pixels,
            EPS,
        )
    )

    channel_location = np.median(
        log_pixels,
        axis=0,
    )

    # Green is the anchor.
    gains = np.exp(
        channel_location[1]
        - channel_location
    )

    return normalize_rgb_gains(
        gains
    )


# ============================================================
# FLOAT LAB NEUTRALITY SCORE
# ============================================================


def neutral_lab_score(
    pixels_rgb: np.ndarray,
    gains: Sequence[float],
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Measure neutral-patch quality in floating-point LAB.

    Returns:

        total_score
        median_chroma
        median_a
        median_b
    """

    if pixels_rgb.size == 0:
        return (
            1e9,
            100.0,
            100.0,
            100.0,
        )

    pixels = np.asarray(
        pixels_rgb,
        dtype=np.float32,
    )

    if pixels.ndim != 2:
        pixels = pixels.reshape(
            -1,
            3,
        )

    normalized = normalize_rgb_gains(
        gains
    )

    corrected = np.clip(
        pixels
        * normalized.reshape(
            1,
            3,
        ),
        0.0,
        1.0,
    )

    lab = _rgb_to_lab_float(
        corrected.reshape(
            -1,
            1,
            3,
        )
    ).reshape(
        -1,
        3,
    )

    a = lab[
        :, 1
    ]

    b = lab[
        :, 2
    ]

    chroma = np.sqrt(
        a * a
        + b * b
    )

    median_a = robust_median(
        a,
        0.0,
    )

    median_b = robust_median(
        b,
        0.0,
    )

    median_chroma = robust_median(
        chroma,
        100.0,
    )

    cast_distance = math.sqrt(
        median_a * median_a
        + median_b * median_b
    )

    movement = (
        abs(
            safe_log(
                float(
                    normalized[0]
                )
            )
        )
        + abs(
            safe_log(
                float(
                    normalized[2]
                )
            )
        )
    )

    # Neutrality is primary.
    #
    # Correction magnitude is only a small regularizer.
    score = (
        median_chroma
        + 1.25 * cast_distance
        + 0.22 * movement
    )

    return (
        float(score),
        float(median_chroma),
        float(median_a),
        float(median_b),
    )


# ============================================================
# NEUTRAL WB OPTIMIZER
# ============================================================


def optimize_neutral_lab_gains(
    rgb: np.ndarray,
    neutral_mask: np.ndarray,
    initial_gains: Optional[
        Sequence[float]
    ] = None,
) -> np.ndarray:
    """
    Find R/B gains that minimize floating-point LAB neutrality.

    Uses deterministic coarse-to-fine optimization.

    Important:
    The objective is regularized toward identity so that a small
    textured gray region cannot cause a huge WB correction.
    """

    mask = normalized_mask(
        neutral_mask
    )

    if mask is None:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = rgb[
        mask
    ]

    if pixels.shape[0] < MIN_NEUTRAL_PIXELS:
        return np.ones(
            3,
            dtype=np.float32,
        )

    valid = (
        np.all(
            np.isfinite(
                pixels
            ),
            axis=1,
        )
        & (
            pixels.min(axis=1)
            > 0.025
        )
        & (
            pixels.max(axis=1)
            < 0.975
        )
    )

    pixels = pixels[
        valid
    ]

    if pixels.shape[0] < MIN_NEUTRAL_PIXELS:
        return np.ones(
            3,
            dtype=np.float32,
        )

    pixels = _sample_pixels(
        pixels,
        MAX_NEUTRAL_OPT_PIXELS,
    )

    if initial_gains is None:

        initial = neutral_patch_gains(
            rgb,
            mask,
        )

    else:

        initial = normalize_rgb_gains(
            initial_gains
        )

    best = normalize_rgb_gains(
        initial
    )

    base_score, _, _, _ = neutral_lab_score(
        pixels,
        best,
    )

    # --------------------------------------------------------
    # Search in logarithmic gain space.
    # --------------------------------------------------------

    def objective(
        candidate: np.ndarray,
    ) -> float:

        score, _, _, _ = neutral_lab_score(
            pixels,
            candidate,
        )

        # Small regularization against unnecessarily large WB.
        distance = _gain_distance_from_identity(
            candidate
        )

        return (
            score
            + 0.18 * distance
        )

    best_score = objective(
        best
    )

    # Coarse search.
    coarse = np.linspace(
        -0.20,
        0.20,
        9,
        dtype=np.float32,
    )

    for r_offset in coarse:

        for b_offset in coarse:

            candidate = normalize_rgb_gains(
                [
                    math.exp(
                        safe_log(
                            float(
                                best[0]
                            )
                        )
                        + float(
                            r_offset
                        )
                    ),
                    1.0,
                    math.exp(
                        safe_log(
                            float(
                                best[2]
                            )
                        )
                        + float(
                            b_offset
                        )
                    ),
                ]
            )

            score = objective(
                candidate
            )

            if score < best_score:
                best_score = score
                best = candidate

    # Fine search.
    fine = np.linspace(
        -0.045,
        0.045,
        9,
        dtype=np.float32,
    )

    for r_offset in fine:

        for b_offset in fine:

            candidate = normalize_rgb_gains(
                [
                    math.exp(
                        safe_log(
                            float(
                                best[0]
                            )
                        )
                        + float(
                            r_offset
                        )
                    ),
                    1.0,
                    math.exp(
                        safe_log(
                            float(
                                best[2]
                            )
                        )
                        + float(
                            b_offset
                        )
                    ),
                ]
            )

            score = objective(
                candidate
            )

            if score < best_score:
                best_score = score
                best = candidate

    # Never return an objectively worse correction.
    if best_score > objective(
        normalize_rgb_gains(
            initial
        )
    ):
        return normalize_rgb_gains(
            initial
        )

    return normalize_rgb_gains(
        best
    )


# ============================================================
# LEARNING BASED WB
# ============================================================


def learning_based_wb_gains(
    image: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Optional OpenCV xphoto LearningBasedWB estimator.

    This estimator is treated as a secondary opinion.

    It must NEVER automatically override a strong neutral reference.
    """

    try:

        if not hasattr(
            cv2,
            "xphoto",
        ):
            return None

        if not hasattr(
            cv2.xphoto,
            "createLearningBasedWB",
        ):
            return None

        wb = cv2.xphoto.createLearningBasedWB()

        corrected = wb.balanceWhite(
            image
        )

        original_rgb = (
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            ).astype(
                np.float32
            )
            / 255.0
        )

        corrected_rgb = (
            cv2.cvtColor(
                corrected,
                cv2.COLOR_BGR2RGB,
            ).astype(
                np.float32
            )
            / 255.0
        )

        valid = (
            np.all(
                np.isfinite(
                    original_rgb
                ),
                axis=2,
            )
            & np.all(
                np.isfinite(
                    corrected_rgb
                ),
                axis=2,
            )
            & (
                original_rgb.min(
                    axis=2
                )
                > 0.025
            )
            & (
                original_rgb.max(
                    axis=2
                )
                < 0.98
            )
            & (
                corrected_rgb.min(
                    axis=2
                )
                > 0.02
            )
            & (
                corrected_rgb.max(
                    axis=2
                )
                < 0.99
            )
        )

        if np.count_nonzero(
            valid
        ) < 200:

            return None

        src = original_rgb[
            valid
        ]

        dst = corrected_rgb[
            valid
        ]

        src = _sample_pixels(
            src,
            30_000,
        )

        # Use the corresponding approximate samples from the
        # beginning rather than introducing nondeterministic sampling.
        dst = dst[
            : src.shape[0]
        ]

        ratios = (
            dst
            / np.maximum(
                src,
                EPS,
            )
        )

        # Median is safer than mean.
        gains = np.median(
            ratios,
            axis=0,
        )

        if not np.all(
            np.isfinite(
                gains
            )
        ):
            return None

        return normalize_rgb_gains(
            gains
        )

    except Exception as exc:

        logger.debug(
            "LearningBasedWB unavailable: %s",
            exc,
        )

        return None


# ============================================================
# WB CANDIDATE EVALUATION
# ============================================================


def evaluate_wb_candidate(
    rgb: np.ndarray,
    neutral_mask: Optional[np.ndarray],
    gains: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a WB candidate against a neutral reference.
    """

    if neutral_mask is None:

        return {
            "before": 100.0,
            "residual": 100.0,
            "improvement": 0.0,
            "cast_distance": 100.0,
        }

    mask = normalized_mask(
        neutral_mask
    )

    if mask is None:

        return {
            "before": 100.0,
            "residual": 100.0,
            "improvement": 0.0,
            "cast_distance": 100.0,
        }

    if np.count_nonzero(
        mask
    ) < MIN_NEUTRAL_PIXELS:

        return {
            "before": 100.0,
            "residual": 100.0,
            "improvement": 0.0,
            "cast_distance": 100.0,
        }

    pixels = rgb[
        mask
    ]

    pixels = _sample_pixels(
        pixels,
        MAX_NEUTRAL_OPT_PIXELS,
    )

    identity = np.ones(
        3,
        dtype=np.float32,
    )

    before_score, before_chroma, before_a, before_b = (
        neutral_lab_score(
            pixels,
            identity,
        )
    )

    after_score, after_chroma, after_a, after_b = (
        neutral_lab_score(
            pixels,
            gains,
        )
    )

    improvement = (
        before_score
        - after_score
    ) / max(
        before_score,
        EPS,
    )

    before_cast = math.sqrt(
        before_a * before_a
        + before_b * before_b
    )

    after_cast = math.sqrt(
        after_a * after_a
        + after_b * after_b
    )

    return {
        "before": float(
            before_score
        ),
        "residual": float(
            after_score
        ),
        "improvement": float(
            improvement
        ),
        "cast_distance": float(
            after_cast
        ),
        "before_chroma": float(
            before_chroma
        ),
        "after_chroma": float(
            after_chroma
        ),
        "before_cast": float(
            before_cast
        ),
        "after_cast": float(
            after_cast
        ),
    }


# ============================================================
# ESTIMATOR AGREEMENT
# ============================================================


def calculate_estimator_agreement(
    estimators: Sequence[
        Mapping[str, Any]
    ],
) -> float:
    """
    Calculate agreement in log-gain space.

    Log space is appropriate because WB corrections are multiplicative.
    """

    if len(
        estimators
    ) < 2:

        return 0.0

    matrix = np.asarray(
        [
            normalize_rgb_gains(
                item["gains"]
            )
            for item in estimators
        ],
        dtype=np.float32,
    )

    if matrix.shape[0] < 2:
        return 0.0

    log_matrix = np.log(
        np.maximum(
            matrix[:, [0, 2]],
            EPS,
        )
    )

    spread = float(
        np.mean(
            np.std(
                log_matrix,
                axis=0,
            )
        )
    )

    # Approximate:
    #
    # spread <= 0.02  -> very strong agreement
    # spread ~= 0.10   -> weak agreement
    # spread >= 0.16   -> disagreement
    agreement = clamp(
        1.0
        - spread / 0.16,
        0.0,
        1.0,
    )

    return float(
        agreement
    )


# ============================================================
# CONSENSUS WB
# ============================================================


def _weighted_log_consensus(
    estimators: Sequence[
        Mapping[str, Any]
    ],
) -> np.ndarray:
    """
    Combine WB estimators in log-gain space.
    """

    if not estimators:
        return np.ones(
            3,
            dtype=np.float32,
        )

    weights = []

    gains = []

    for item in estimators:

        gain = normalize_rgb_gains(
            item["gains"]
        )

        base_weight = clamp(
            safe_float(
                item.get(
                    "base_weight",
                    0.1,
                ),
                0.1,
            ),
            0.0,
            1.0,
        )

        improvement = clamp(
            safe_float(
                item.get(
                    "improvement",
                    0.0,
                ),
                0.0,
            ),
            0.0,
            1.0,
        )

        # An estimator that actually improves the neutral reference
        # gets more influence.
        evidence = (
            0.25
            + 0.75 * improvement
        )

        weight = (
            base_weight
            * evidence
        )

        gains.append(
            gain
        )

        weights.append(
            weight
        )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    if float(
        np.sum(weights)
    ) <= EPS:

        return np.ones(
            3,
            dtype=np.float32,
        )

    weights /= np.sum(
        weights
    )

    log_result = np.zeros(
        3,
        dtype=np.float32,
    )

    for gain, weight in zip(
        gains,
        weights,
    ):

        log_result += (
            np.log(
                np.maximum(
                    gain,
                    EPS,
                )
            )
            * float(weight)
        )

    return normalize_rgb_gains(
        np.exp(
            log_result
        )
    )


# ============================================================
# WHITE BALANCE ANALYSIS
# ============================================================


def _analyze_white_balance_neutral(
    image: np.ndarray,
    semantic: Mapping[
        str,
        Optional[np.ndarray],
    ],
    subject: Optional[np.ndarray],
    use_learning_wb: bool = DEFAULT_USE_LEARNING_WB,
) -> Dict[str, Any]:
    """
    Main WB analyzer.

    Decision hierarchy:

        1. Strong neutral reference
        2. Neutral optimizer
        3. Secondary estimators
        4. Consensus only when no neutral exists
        5. Identity/no-op if evidence is insufficient
    """

    if image is None:
        raise ValueError(
            "image is None"
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be numpy.ndarray"
        )

    if image.ndim != 3:
        raise ValueError(
            "image must be HxWx3"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "image must contain 3 channels"
        )

    if image.dtype != np.uint8:
        raise TypeError(
            "image must be uint8"
        )

    rgb = bgr_to_rgb_float(
        image
    )

    # ========================================================
    # NEUTRAL PATCH
    # ========================================================

    neutral = find_neutral_patches(
        image,
        semantic,
        subject,
    )

    neutral_mask = (
        neutral.mask
        if neutral.count >= MIN_NEUTRAL_PIXELS
        else None
    )

    # ========================================================
    # GLOBAL VALID PIXELS
    # ========================================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    valid = (
        (hsv[:, :, 1] < 215)
        & (image.min(axis=2) > 7)
        & (image.max(axis=2) < 248)
    )

    excluded = _build_neutral_exclusion(
        semantic,
        image.shape[:2],
    )

    valid &= ~excluded

    # ========================================================
    # ESTIMATORS
    # ========================================================

    estimators: list[
        Dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Neutral reference
    # --------------------------------------------------------

    neutral_initial: Optional[
        np.ndarray
    ] = None

    if neutral_mask is not None:

        neutral_initial = (
            neutral_patch_gains(
                rgb,
                neutral_mask,
            )
        )

        neutral_optimized = (
            optimize_neutral_lab_gains(
                rgb,
                neutral_mask,
                initial_gains=neutral_initial,
            )
        )

        validation = evaluate_wb_candidate(
            rgb,
            neutral_mask,
            neutral_optimized,
        )

        estimators.append(
            {
                "name": "neutral_lab_optimizer",
                "gains": neutral_optimized,
                "base_weight": 1.00,
                "residual": validation[
                    "residual"
                ],
                "improvement": validation[
                    "improvement"
                ],
                "before": validation[
                    "before"
                ],
                "cast_distance": validation[
                    "cast_distance"
                ],
            }
        )

    # --------------------------------------------------------
    # Learning based
    # --------------------------------------------------------

    if use_learning_wb:

        learning = (
            learning_based_wb_gains(
                image
            )
        )

        if learning is not None:

            validation = evaluate_wb_candidate(
                rgb,
                neutral_mask,
                learning,
            )

            estimators.append(
                {
                    "name": "learning_based",
                    "gains": normalize_rgb_gains(
                        learning
                    ),
                    "base_weight": 0.16,
                    "residual": validation[
                        "residual"
                    ],
                    "improvement": validation[
                        "improvement"
                    ],
                    "before": validation[
                        "before"
                    ],
                    "cast_distance": validation[
                        "cast_distance"
                    ],
                }
            )

    # --------------------------------------------------------
    # Shades of gray
    # --------------------------------------------------------

    shades = shades_of_gray_gains(
        rgb,
        p=6.0,
        valid=valid,
    )

    validation = evaluate_wb_candidate(
        rgb,
        neutral_mask,
        shades,
    )

    estimators.append(
        {
            "name": "shades_of_gray",
            "gains": normalize_rgb_gains(
                shades
            ),
            "base_weight": 0.14,
            "residual": validation[
                "residual"
            ],
            "improvement": validation[
                "improvement"
            ],
            "before": validation[
                "before"
            ],
            "cast_distance": validation[
                "cast_distance"
            ],
        }
    )

    # --------------------------------------------------------
    # Gray world
    # --------------------------------------------------------

    gray = gray_world_gains(
        rgb,
        valid=valid,
    )

    validation = evaluate_wb_candidate(
        rgb,
        neutral_mask,
        gray,
    )

    estimators.append(
        {
            "name": "gray_world",
            "gains": normalize_rgb_gains(
                gray
            ),
            "base_weight": 0.08,
            "residual": validation[
                "residual"
            ],
            "improvement": validation[
                "improvement"
            ],
            "before": validation[
                "before"
            ],
            "cast_distance": validation[
                "cast_distance"
            ],
        }
    )

    # ========================================================
    # AGREEMENT
    # ========================================================

    agreement = calculate_estimator_agreement(
        estimators
    )

    # ========================================================
    # SELECT PRIMARY CORRECTION
    # ========================================================

    method = (
        "identity_insufficient_evidence"
    )

    fused = np.ones(
        3,
        dtype=np.float32,
    )

    raw_confidence = 0.0

    # --------------------------------------------------------
    # Case 1:
    # Strong neutral reference
    # --------------------------------------------------------

    if (
        neutral_mask is not None
        and neutral.count >= MIN_NEUTRAL_PIXELS
        and estimators
    ):

        neutral_estimator = next(
            (
                item
                for item in estimators
                if item["name"]
                == "neutral_lab_optimizer"
            ),
            None,
        )

        if neutral_estimator is not None:

            neutral_gains = normalize_rgb_gains(
                neutral_estimator[
                    "gains"
                ]
            )

            neutral_improvement = clamp(
                safe_float(
                    neutral_estimator.get(
                        "improvement",
                        0.0,
                    )
                ),
                0.0,
                1.0,
            )

            # Stronger neutral confidence means we can trust
            # the neutral reference more heavily.
            neutral_quality = clamp(
                neutral.confidence,
                0.0,
                1.0,
            )

            # If the patch is genuinely neutral already,
            # do not invent a correction.
            if (
                neutral_improvement
                < MIN_USEFUL_IMPROVEMENT
            ):

                fused = np.ones(
                    3,
                    dtype=np.float32,
                )

                method = (
                    "identity_neutral_already"
                )

                raw_confidence = (
                    0.45
                    * neutral_quality
                    + 0.20
                    * agreement
                )

            else:

                # ------------------------------------------------
                # Secondary estimators can only influence the
                # neutral result slightly.
                # ------------------------------------------------

                secondary = [
                    item
                    for item in estimators
                    if item["name"]
                    != "neutral_lab_optimizer"
                ]

                secondary_consensus = (
                    _weighted_log_consensus(
                        secondary
                    )
                    if secondary
                    else np.ones(
                        3,
                        dtype=np.float32,
                    )
                )

                # Neutral patch remains dominant.
                neutral_log = np.log(
                    np.maximum(
                        neutral_gains,
                        EPS,
                    )
                )

                secondary_log = np.log(
                    np.maximum(
                        secondary_consensus,
                        EPS,
                    )
                )

                # At most 15% secondary influence.
                secondary_weight = (
                    0.04
                    + 0.11 * agreement
                )

                fused_log = (
                    neutral_log
                    * (
                        1.0
                        - secondary_weight
                    )
                    + secondary_log
                    * secondary_weight
                )

                fused = normalize_rgb_gains(
                    np.exp(
                        fused_log
                    )
                )

                method = (
                    "neutral_reference_dominant"
                )

                raw_confidence = (
                    0.55
                    * neutral_quality
                    + 0.25
                    * clamp(
                        neutral_improvement
                        * 1.6,
                        0.0,
                        1.0,
                    )
                    + 0.20
                    * agreement
                )

    # --------------------------------------------------------
    # Case 2:
    # No reliable neutral patch
    # --------------------------------------------------------

    elif len(
        estimators
    ) >= 2:

        if agreement >= 0.72:

            fused = (
                _weighted_log_consensus(
                    estimators
                )
            )

            # Without a neutral reference, restrict correction.
            log_fused = np.log(
                np.maximum(
                    fused,
                    EPS,
                )
            )

            log_fused[0] = clamp(
                float(
                    log_fused[0]
                ),
                -0.16,
                0.16,
            )

            log_fused[2] = clamp(
                float(
                    log_fused[2]
                ),
                -0.16,
                0.16,
            )

            fused = normalize_rgb_gains(
                np.exp(
                    log_fused
                )
            )

            method = (
                "multi_estimator_consensus"
            )

            raw_confidence = (
                0.70
                * agreement
            )

        else:

            fused = np.ones(
                3,
                dtype=np.float32,
            )

            method = (
                "identity_estimators_disagree"
            )

            raw_confidence = (
                0.18
                * agreement
            )

    # ========================================================
    # FINAL CONFIDENCE
    # ========================================================

    raw_confidence = clamp(
        raw_confidence,
        0.0,
        0.99,
    )

    # Identity decisions should never claim high confidence.
    if method.startswith(
        "identity"
    ):
        raw_confidence = min(
            raw_confidence,
            0.35,
        )

    # ========================================================
    # FINAL WB SAFETY
    # ========================================================

    final_gains = _finalize_wb_gains(
        fused,
        raw_confidence,
    )

    # --------------------------------------------------------
    # Final no-op gate.
    #
    # Prevents microscopic numerical WB changes.
    # --------------------------------------------------------

    gain_distance = (
        _gain_distance_from_identity(
            final_gains
        )
    )

    if gain_distance < NOOP_LOG_THRESHOLD:

        final_gains = np.ones(
            3,
            dtype=np.float32,
        )

        if not method.startswith(
            "identity"
        ):
            method = (
                "identity_correction_too_small"
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if neutral_mask is not None:

        final_validation = (
            evaluate_wb_candidate(
                rgb,
                neutral_mask,
                final_gains,
            )
        )

        before = safe_float(
            final_validation.get(
                "before",
                100.0,
            ),
            100.0,
        )

        after = safe_float(
            final_validation.get(
                "residual",
                100.0,
            ),
            100.0,
        )

        improvement = clamp(
            (
                before - after
            )
            / max(
                before,
                EPS,
            ),
            0.0,
            1.0,
        )

    else:

        before = 100.0
        after = 100.0
        improvement = 0.0

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    # If a neutral reference exists and our final correction
    # doesn't actually improve it, reject the correction.
    if (
        neutral_mask is not None
        and improvement
        < MIN_USEFUL_IMPROVEMENT
        and gain_distance
        > NOOP_LOG_THRESHOLD
    ):

        final_gains = np.ones(
            3,
            dtype=np.float32,
        )

        method = (
            "identity_final_validation_failed"
        )

        raw_confidence = min(
            raw_confidence,
            0.25,
        )

        after = before
        improvement = 0.0

    # ========================================================
    # CAST METRICS
    # ========================================================

    temperature, tint = (
        gains_to_temperature_tint(
            final_gains
        )
    )

    if abs(
        temperature
    ) < 0.5:

        temperature = 0.0

    if abs(
        tint
    ) < 0.5:

        tint = 0.0

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "gains_rgb": [
            round(
                float(
                    final_gains[0]
                ),
                6,
            ),
            1.0,
            round(
                float(
                    final_gains[2]
                ),
                6,
            ),
        ],

        "temperature_cast": round(
            float(
                temperature
            ),
            4,
        ),

        "tint_cast": round(
            float(
                tint
            ),
            4,
        ),

        "confidence": round(
            float(
                raw_confidence
            ),
            4,
        ),

        "neutral_patch_count": int(
            neutral.count
        ),

        "neutral_coverage": round(
            float(
                neutral.coverage
            ),
            6,
        ),

        "neutrality_before": round(
            float(
                before
            ),
            4,
        ),

        "neutrality_after": round(
            float(
                after
            ),
            4,
        ),

        "neutrality_improvement_percentage": round(
            float(
                improvement
                * 100.0
            ),
            3,
        ),

        "neutral_patch_confidence": round(
            float(
                neutral.confidence
            ),
            4,
        ),

        "neutral_patch_spatial_quality": round(
            float(
                neutral.spatial_quality
            ),
            4,
        ),

        "method": method,

        "estimator_agreement": round(
            float(
                agreement
            ),
            5,
        ),

        "wb_correction_strength": round(
            float(
                gain_distance
            ),
            6,
        ),

        "learning_wb_enabled": bool(
            use_learning_wb
        ),

        "estimators": [
            {
                "name": item[
                    "name"
                ],

                "gains_rgb": [
                    round(
                        float(
                            x
                        ),
                        6,
                    )
                    for x in normalize_rgb_gains(
                        item[
                            "gains"
                        ]
                    )
                ],

                "weight": round(
                    float(
                        item[
                            "base_weight"
                        ]
                    ),
                    5,
                ),

                "neutrality_before": round(
                    float(
                        item.get(
                            "before",
                            100.0,
                        )
                    ),
                    4,
                ),

                "neutrality_after": round(
                    float(
                        item.get(
                            "residual",
                            100.0,
                        )
                    ),
                    4,
                ),

                "improvement": round(
                    float(
                        item.get(
                            "improvement",
                            0.0,
                        )
                    ),
                    5,
                ),

                "cast_distance": round(
                    float(
                        item.get(
                            "cast_distance",
                            100.0,
                        )
                    ),
                    4,
                ),
            }
            for item in estimators
        ],
    }


# ============================================================
# PUBLIC API
# ============================================================


def analyze_white_balance(
    image: np.ndarray,
    semantic: Mapping[
        str,
        Optional[np.ndarray],
    ],
    subject: Optional[np.ndarray],
    use_learning_wb: bool = DEFAULT_USE_LEARNING_WB,
) -> Dict[str, Any]:
    """
    Public white-balance analysis API.

    Parameters
    ----------
    image:
        BGR uint8 image.

    semantic:
        Semantic masks dictionary.

    subject:
        Optional subject/person mask.

    use_learning_wb:
        Enable OpenCV xphoto LearningBasedWB.

    Returns
    -------
    dict
        JSON-compatible WB correction plan.
    """

    return _analyze_white_balance_neutral(
        image=image,
        semantic=semantic,
        subject=subject,
        use_learning_wb=use_learning_wb,
    )