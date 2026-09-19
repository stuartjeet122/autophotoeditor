from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from autophotoeditor.core.image_io import decode_color


logger = logging.getLogger("auto_enhance")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class EnhancementConfig:
    """
    Natural adaptive photographic enhancement.

    Design goals:

        - preserve photographic intent
        - avoid unnecessary warm/cool shifts
        - avoid forced histogram normalization
        - preserve midtones and skin
        - recover usable shadows gently
        - roll highlights instead of turning them gray
        - avoid excessive clarity/sharpening
    """

    # ------------------------------------------------------------------------
    # WHITE BALANCE
    # ------------------------------------------------------------------------

    enable_wb: bool = True

    # Base WB strength.
    wb_strength: float = 1.00

    # Conservative correction limits.
    wb_min_scale: float = 0.75
    wb_max_scale: float = 1.35

    # Stronger limits when confidence is exceptionally high.
    wb_high_conf_min_scale: float = 0.75
    wb_high_conf_max_scale: float = 1.35

    # Minimum number of pixels used for neutral estimation.
    wb_min_neutral_pixels: int = 150

    # Minimum percentage of image that must provide neutral evidence.
    wb_min_coverage: float = 0.008

    # Ignore tiny measured casts.
    wb_temperature_deadzone: float = 0.35
    wb_tint_deadzone: float = 0.35

    # Do not aggressively neutralize strong scene illumination.
    wb_strong_cast_threshold: float = 10.0

    # ------------------------------------------------------------------------
    # EXPOSURE
    # ------------------------------------------------------------------------

    enable_tone: bool = True

    # Reference midpoint, but NOT a hard target.
    exposure_reference: float = 121.0

    # Maximum exposure movement.
    max_exposure_shift: float = 28.0

    # Overall exposure response.
    exposure_strength: float = 0.46

    # ------------------------------------------------------------------------
    # ADAPTIVE CONTRAST
    # ------------------------------------------------------------------------

    contrast_strength: float = 0.32

    # Percentiles are used for diagnosis rather than hard histogram forcing.
    low_percentile: float = 2.0
    high_percentile: float = 98.0

    # ------------------------------------------------------------------------
    # HIGHLIGHTS
    # ------------------------------------------------------------------------

    enable_highlight_protection: bool = True

    highlight_start: float = 218.0
    highlight_roll_start: float = 235.0
    highlight_roll_end: float = 255.0

    highlight_strength: float = 0.24

    # ------------------------------------------------------------------------
    # SHADOWS
    # ------------------------------------------------------------------------

    enable_shadow_protection: bool = True

    shadow_start: float = 8.0
    shadow_open_end: float = 55.0

    shadow_strength: float = 0.12

    # ------------------------------------------------------------------------
    # CLARITY
    # ------------------------------------------------------------------------

    enable_clarity: bool = True

    clarity_amount: float = 0.12
    clarity_max_amount: float = 0.24

    clarity_sigma: float = 18.0

    # ------------------------------------------------------------------------
    # VIBRANCE
    # ------------------------------------------------------------------------

    enable_vibrance: bool = True

    vibrance_low_sat: float = 1.18
    vibrance_medium_sat: float = 1.10
    vibrance_high_sat: float = 1.05
    vibrance_already_high: float = 1.015

    saturation_limit: int = 242

    # ------------------------------------------------------------------------
    # SHARPENING
    # ------------------------------------------------------------------------

    enable_sharpen: bool = True

    sharpness_soft_threshold: float = 80.0
    sharpness_good_threshold: float = 220.0
    sharpness_high_threshold: float = 500.0

    sharp_sigma: float = 0.85
    sharp_amount: float = 0.22
    sharp_max_amount: float = 0.34

    # ------------------------------------------------------------------------
    # NOISE
    # ------------------------------------------------------------------------

    noise_low: float = 3.0
    noise_medium: float = 7.0
    noise_high: float = 12.0

    # ------------------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------------------

    max_total_luma_change: float = 72.0


class EnhancementError(Exception):
    """Raised when enhancement cannot proceed."""


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except Exception:
        return default


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


def _smoothstep(
    x: np.ndarray,
) -> np.ndarray:

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return (
        x *
        x *
        (3.0 - 2.0 * x)
    )


def _smoothstep_scalar(
    x: float,
) -> float:

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x *
        x *
        (3.0 - 2.0 * x)
    )


# ============================================================================
# ANALYSIS ACCESSORS
# ============================================================================

def _get_exposure(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    value = analysis.get("exposure_and_tone", {})
    return value if isinstance(value, dict) else {}


def _get_recommendations(analysis: dict[str, Any]) -> dict[str, Any]:
    value = analysis.get("recommendations", {})
    return value if isinstance(value, dict) else {}


def _get_scene(analysis: dict[str, Any]) -> dict[str, Any]:
    value = analysis.get("scene", {})
    return value if isinstance(value, dict) else {}


def _get_v3_white_balance(analysis: dict[str, Any]) -> dict[str, Any]:
    value = analysis.get("white_balance", {})
    return value if isinstance(value, dict) else {}


def _get_color(
    analysis: dict[str, Any],
) -> dict[str, Any]:

    value = analysis.get(
        "color_profile",
        {},
    )

    return value if isinstance(value, dict) else {}


def _get_quality(
    analysis: dict[str, Any],
) -> dict[str, Any]:

    value = analysis.get(
        "quality",
        {},
    )

    return value if isinstance(value, dict) else {}


def _get_median_luminance(
    exposure: dict[str, Any],
) -> float:

    if "median_luminance" in exposure:
        return _safe_float(
            exposure["median_luminance"],
            118.0,
        )

    if "brightness_luminance" in exposure:
        return _safe_float(
            exposure["brightness_luminance"],
            118.0,
        )

    return 118.0


def _get_mean_luminance(
    exposure: dict[str, Any],
) -> float:

    if "mean_luminance" in exposure:
        return _safe_float(
            exposure["mean_luminance"],
            118.0,
        )

    if "brightness_luminance" in exposure:
        return _safe_float(
            exposure["brightness_luminance"],
            118.0,
        )

    return 118.0


def _get_saturation(
    color: dict[str, Any],
) -> float:

    return _safe_float(
        color.get(
            "saturation_mean",
            70.0,
        ),
        70.0,
    )


def _get_sharpness(
    quality: dict[str, Any],
) -> float:

    if "laplacian_variance" in quality:
        return _safe_float(
            quality["laplacian_variance"],
            100.0,
        )

    if "sharpness_variance" in quality:
        return _safe_float(
            quality["sharpness_variance"],
            100.0,
        )

    return 100.0


def _get_noise(
    quality: dict[str, Any],
) -> float:

    return _safe_float(
        quality.get(
            "noise_estimate_std",
            0.0,
        ),
        0.0,
    )


# ============================================================================
# COLOR / LUMINANCE HELPERS
# ============================================================================

def _lab_luminance(
    img: np.ndarray,
) -> np.ndarray:

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    return lab[:, :, 0]


def _calculate_luma(
    img: np.ndarray,
) -> np.ndarray:

    b = img[:, :, 0].astype(
        np.float32
    )

    g = img[:, :, 1].astype(
        np.float32
    )

    r = img[:, :, 2].astype(
        np.float32
    )

    return (
        0.114 * b +
        0.587 * g +
        0.299 * r
    )


# ============================================================================ #
# WHITE BALANCE - NATURAL NEUTRALIZATION
# ============================================================================ #

def _find_neutral_pixels(
    img: np.ndarray,
) -> np.ndarray:
    """
    Find reliable neutral reference pixels.

    We intentionally make this more selective than a simple low-saturation
    mask. Neutral pixels should have:

        - useful brightness
        - low saturation
        - low Lab chroma
        - reasonably similar RGB channels
        - no extreme shadows/highlights

    This prevents dark foliage, colored walls, skin, etc. from becoming
    accidental white-balance references.
    """

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV,
    )

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # Lab chroma distance from neutral.
    chroma = np.sqrt(
        (A - 128.0) ** 2 +
        (B - 128.0) ** 2
    )

    blue = img[:, :, 0].astype(np.float32)
    green = img[:, :, 1].astype(np.float32)
    red = img[:, :, 2].astype(np.float32)

    rgb_mean = (
        blue +
        green +
        red
    ) / 3.0

    rgb_max = np.maximum.reduce(
        [blue, green, red]
    )

    rgb_min = np.minimum.reduce(
        [blue, green, red]
    )

    rgb_spread = (
        rgb_max -
        rgb_min
    )

    relative_spread = (
        rgb_spread /
        np.maximum(
            rgb_mean,
            12.0,
        )
    )

    # --------------------------------------------------------------------- #
    # Main neutral test.
    # --------------------------------------------------------------------- #

    # The old thresholds were too strict for globally cast but otherwise neutral
    # scenes. A warm or cool image can still contain valid neutral references even
    # when the entire frame shares the same tint. We keep the selection narrow,
    # but allow a realistic amount of chroma and channel spread from the cast.
    mask = (
        (L >= 30.0) &
        (L <= 240.0) &
        (V >= 35.0) &
        (V <= 245.0) &
        (S <= 52.0) &
        (chroma <= 18.0) &
        (relative_spread <= 0.42)
    )

    # --------------------------------------------------------------------- #
    # Reject pixels that are too close to black/white.
    #
    # Those pixels are poor WB references because clipping destroys
    # chromatic information.
    # --------------------------------------------------------------------- #

    mask &= (
        (rgb_mean >= 45.0) &
        (rgb_mean <= 235.0)
    )

    return mask


def _estimate_neutral_rgb(
    img: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    """
    Robustly estimate the neutral RGB reference.

    Median is useful, but using only one median can still be influenced
    by a cluster of slightly warm/cool pixels.

    We therefore:
        1. collect candidate pixels
        2. remove extreme candidates
        3. calculate a robust median
    """

    count = int(
        np.count_nonzero(mask)
    )

    if count < 1:
        return None, 0

    pixels = img[mask].astype(
        np.float32
    )

    if pixels.shape[0] > 5000:

        # Subsample for speed on very large photographs.
        step = max(
            pixels.shape[0] // 5000,
            1,
        )

        pixels = pixels[::step]

    # --------------------------------------------------------------------- #
    # RGB chromaticity.
    # --------------------------------------------------------------------- #

    mean_rgb = np.mean(
        pixels,
        axis=1,
        keepdims=True,
    )

    chromatic_deviation = np.abs(
        pixels -
        mean_rgb
    ).mean(
        axis=1
    )

    # Remove the most suspicious 15%.
    threshold = np.percentile(
        chromatic_deviation,
        85.0,
    )

    good = (
        chromatic_deviation <=
        threshold
    )

    pixels = pixels[good]

    if pixels.shape[0] < 10:
        return None, count

    # Median is robust against residual contamination.
    neutral = np.median(
        pixels,
        axis=0,
    )

    return neutral.astype(
        np.float32
    ), count


def _normalize_wb_gains(
    gains: np.ndarray,
) -> np.ndarray:
    """
    Normalize RGB gains so WB changes chroma but approximately preserves
    luminance.
    """

    gains = np.asarray(
        gains,
        dtype=np.float32,
    )

    luminance_gain = (
        0.114 * gains[0] +
        0.587 * gains[1] +
        0.299 * gains[2]
    )

    if luminance_gain > 1e-6:

        gains = (
            gains /
            luminance_gain
        )

    return gains


def _calculate_wb_confidence(
    temperature: float,
    tint: float,
    coverage: float,
    neutral_count: int,
    image_size: int,
) -> float:
    """
    Estimate confidence that the detected cast should be corrected.

    Unlike the previous implementation, a stronger cast does not
    automatically reduce confidence.

    Strong cast + strong neutral evidence = strong confidence.
    """

    if neutral_count <= 0:
        return 0.0

    coverage_score = _clamp(
        coverage / 0.08,
        0.0,
        1.0,
    )

    count_score = _clamp(
        neutral_count /
        max(
            image_size * 0.03,
            1.0,
        ),
        0.0,
        1.0,
    )

    cast = max(
        abs(temperature),
        abs(tint),
    )

    # A tiny measured cast should not produce a correction.
    cast_score = _clamp(
        cast / 12.0,
        0.0,
        1.0,
    )

    confidence = (
        0.55 * coverage_score +
        0.25 * count_score +
        0.20 * cast_score
    )

    return _clamp(
        confidence,
        0.0,
        1.0,
    )


def _srgb_to_linear_u8(img: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB BGR to linear-light float BGR in [0, 1]."""
    x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert linear-light BGR float [0,1] to uint8 sRGB."""
    x = np.clip(rgb, 0.0, 1.0)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _apply_linear_rgb_gains(img: np.ndarray, gains_bgr: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear_u8(img)
    linear *= np.asarray(gains_bgr, dtype=np.float32)[None, None, :]
    return _linear_to_srgb_u8(linear)


def _apply_white_balance(
    img: np.ndarray,
    analysis: dict[str, Any],
    color: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:
    """Neutralize the measured illuminant in linear RGB.

    Unlike the previous implementation this does not intentionally leave a
    percentage of a detected cast behind.  Confidence decides whether the
    measurement is trustworthy; once trusted, the measured correction is
    applied at full strength in linear-light RGB.
    """
    if not cfg.enable_wb:
        return img

    wb = _get_v3_white_balance(analysis)
    if not wb:
        compat = color.get("white_balance", {})
        wb = compat if isinstance(compat, dict) else {}

    temp_obj = wb.get("estimated_temperature_cast", {})
    tint_obj = wb.get("estimated_tint_cast", {})
    confidence = _safe_float(
        temp_obj.get("confidence", wb.get("confidence", 0.0))
        if isinstance(temp_obj, dict) else wb.get("confidence", 0.0),
        0.0,
    )

    gains_value = wb.get("correction_gains_rgb")
    if isinstance(gains_value, (list, tuple)) and len(gains_value) == 3:
        gains_rgb = np.asarray([_safe_float(x, 1.0) for x in gains_value], dtype=np.float32)
    else:
        temperature = _safe_float(temp_obj.get("value", 0.0) if isinstance(temp_obj, dict) else temp_obj, 0.0)
        tint = _safe_float(tint_obj.get("value", 0.0) if isinstance(tint_obj, dict) else tint_obj, 0.0)
        gains_rgb = np.array([
            1.0 + temperature * 0.0035,
            1.0 - tint * 0.0020,
            1.0 - temperature * 0.0035,
        ], dtype=np.float32)

    if not np.all(np.isfinite(gains_rgb)) or confidence < 0.45:
        return img

    gains_rgb = gains_rgb / max(float(gains_rgb[1]), 1e-6)
    gains_rgb = np.clip(gains_rgb, cfg.wb_min_scale, cfg.wb_max_scale)

    # Preserve luminance while changing chromaticity.
    lum = 0.2126 * gains_rgb[0] + 0.7152 * gains_rgb[1] + 0.0722 * gains_rgb[2]
    if lum > 1e-6:
        gains_rgb /= lum

    # Very small residuals are not worth changing the image.
    if np.max(np.abs(gains_rgb - 1.0)) < 0.0015:
        return img

    gains_bgr = gains_rgb[[2, 1, 0]].astype(np.float32)
    return _apply_linear_rgb_gains(img, gains_bgr)


def _residual_neutral_gain(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure residual WB error from conservative neutral pixels.

    This is deliberately independent of the original analyzer so the final
    pass can verify the actual rendered image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    b, g, r = cv2.split(img.astype(np.float32))
    mean = (b + g + r) / 3.0
    spread = (np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])) / np.maximum(mean, 12.0)
    chroma = np.sqrt((lab[:, :, 1].astype(np.float32)-128.0)**2 + (lab[:, :, 2].astype(np.float32)-128.0)**2)
    mask = (L >= 40) & (L <= 225) & (S <= 42) & (chroma <= 15) & (mean >= 50) & (mean <= 225) & (spread <= 0.32)
    if int(mask.sum()) < 100:
        return np.ones(3, dtype=np.float32), 0.0
    pixels = img[mask].astype(np.float32) / 255.0
    # Use robust medians in linear space.
    x = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    med = np.median(x, axis=0)
    target = float(np.mean(med))
    gains = target / np.maximum(med, 1e-6)
    # BGR -> RGB ordering is irrelevant here because all channels are treated symmetrically.
    gains /= max(float(np.mean(gains)), 1e-6)
    confidence = float(np.clip(mask.mean() / 0.03, 0.0, 1.0))
    return gains.astype(np.float32), confidence


def _apply_residual_white_balance(img: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    gains_bgr, confidence = _residual_neutral_gain(img)
    if confidence < 0.35:
        return img
    gains_bgr = np.clip(gains_bgr, 0.985, 1.015)
    if np.max(np.abs(gains_bgr - 1.0)) < 0.002:
        return img
    return _apply_linear_rgb_gains(img, gains_bgr)


# ============================================================================
# EXPOSURE ESTIMATION
# ============================================================================

def _calculate_exposure_shift(
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> float:
    """Use the v3 analyzer's scene/subject-aware EV recommendation.

    The v3 analyzer deliberately avoids forcing every photograph toward a fixed
    gray midpoint. We therefore use its recommended EV directly and only apply
    conservative safety limits here.
    """
    rec = _get_recommendations(analysis)
    ev = rec.get("exposure_ev", {})
    if isinstance(ev, dict):
        value = _safe_float(ev.get("value", 0.0), 0.0)
        confidence = _safe_float(ev.get("confidence", 0.5), 0.5)
    else:
        value = _safe_float(ev, 0.0)
        confidence = 0.5

    # If v3 recommendations are unavailable, retain compatibility with the
    # older luminance-based analysis.
    if "recommendations" not in analysis:
        median = _get_median_luminance(exposure)
        if median < 55:
            value = (cfg.exposure_reference - median) * 0.48 / 255.0
        elif median < 80:
            value = (cfg.exposure_reference - median) * 0.30 / 255.0
        elif median < 98:
            value = (cfg.exposure_reference - median) * 0.18 / 255.0
        elif median < 140:
            value = 0.0
        elif median < 165:
            value = (cfg.exposure_reference - median) * 0.12 / 255.0
        elif median < 190:
            value = (cfg.exposure_reference - median) * 0.22 / 255.0
        else:
            value = (cfg.exposure_reference - median) * 0.30 / 255.0
        confidence = 0.5

    # Lower-confidence recommendations are blended toward no change.
    confidence = _clamp(confidence, 0.0, 1.0)
    value *= 0.65 + 0.35 * confidence

    # Preserve the existing API's image-space shift. v3 EV is converted to a
    # restrained LAB-L movement; 1 EV is intentionally not treated as 255 L.
    # Approximately 18-24 LAB units gives a visible but controlled result.
    shift = value * 22.0

    clip = exposure.get("clipping", {})
    if isinstance(clip, dict):
        high_clip = _safe_float(clip.get("highlight_all_channels_percentage", 0.0))
        if high_clip > 0.4 and shift > 0:
            shift *= 0.20

    return _clamp(shift, -cfg.max_exposure_shift, cfg.max_exposure_shift)


# ============================================================================
# NATURAL EXPOSURE
# ============================================================================

def _apply_exposure(
    img: np.ndarray,
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:
    """Apply exposure in linear light from the analyzer's EV recommendation."""
    ev = _calculate_exposure_shift(analysis, exposure, cfg) / 22.0
    if abs(ev) < 0.005:
        return img

    # Convert the requested EV to a true linear-light gain.
    gain = float(2.0 ** np.clip(ev * cfg.exposure_strength, -1.5, 1.5))
    linear = _srgb_to_linear_u8(img)

    # Protect highlights progressively when raising exposure.
    y = 0.2126 * linear[:, :, 2] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 0]
    if ev > 0:
        headroom = np.clip((0.96 - y) / 0.30, 0.0, 1.0)
        weight = 0.35 + 0.65 * headroom
        target = linear * gain
        linear = linear * (1.0 - weight[:, :, None]) + target * weight[:, :, None]
    else:
        linear *= gain

    return _linear_to_srgb_u8(linear)


# ============================================================================
# NATURAL CONTRAST CURVE
# ============================================================================

def _build_natural_contrast_lut(
    amount: float,
) -> np.ndarray:
    """
    Build a midpoint-preserving photographic contrast curve.

    Important properties:

        0   remains approximately 0
        128 remains exactly around 128
        255 remains approximately 255

    Contrast is concentrated around the midtones.
    """

    x = np.linspace(
        0.0,
        1.0,
        256,
        dtype=np.float32,
    )

    # Centered sigmoid-like contrast.
    centered = (
        x - 0.5
    )

    # Gentle cubic contrast.
    curve = (
        centered *
        (
            1.0 +
            amount *
            1.65 *
            (
                1.0 -
                4.0 *
                centered *
                centered
            )
        )
    )

    y = (
        0.5 +
        curve
    )

    # Blend curve with identity.
    y = (
        x * (1.0 - amount) +
        y * amount
    )

    return np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)


def _calculate_contrast_amount(
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> float:
    rec = _get_recommendations(analysis)
    if "contrast" in rec:
        value = _safe_float(rec.get("contrast", 0.0), 0.0)
        # v3 contrast is Lightroom-like -100..100. Map conservatively to LUT amount.
        return _clamp(value / 100.0, -0.18, 0.18)

    percentiles = exposure.get("percentiles", {})
    if not isinstance(percentiles, dict):
        percentiles = {}
    p05 = _safe_float(percentiles.get("p5_0", 60.0), 60.0)
    p95 = _safe_float(percentiles.get("p95_0", 200.0), 200.0)
    dynamic_range = _safe_float(exposure.get("effective_dynamic_range", p95 - p05), p95 - p05)
    if dynamic_range < 45:
        amount = cfg.contrast_strength
    elif dynamic_range < 70:
        amount = cfg.contrast_strength * 0.72
    elif dynamic_range < 100:
        amount = cfg.contrast_strength * 0.45
    elif dynamic_range < 140:
        amount = cfg.contrast_strength * 0.22
    else:
        amount = cfg.contrast_strength * 0.08
    return _clamp(amount, 0.0, 0.25)


def _apply_natural_contrast(
    img: np.ndarray,
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    amount = _calculate_contrast_amount(
        analysis,
        exposure,
        cfg,
    )

    if amount < 0.01:
        return img

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    L = lab[:, :, 0]

    lut = _build_natural_contrast_lut(
        amount
    )

    lab[:, :, 0] = cv2.LUT(
        L,
        lut,
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# HIGHLIGHT ROLLOFF
# ============================================================================

def _apply_highlight_rolloff(
    img: np.ndarray,
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    clipping = _safe_float(
        exposure.get(
            "highlight_clipping",
            {},
        ).get(
            "all_channels_percentage",
            0.0,
        )
    )

    bright_pct = _safe_float(
        exposure.get(
            "bright_pixels_percentage",
            0.0,
        )
    )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    L = lab[:, :, 0].astype(
        np.float32
    )

    # Apply to actual bright regions even when the whole image is not clipped.
    if not np.any(L > 200.0):
        return img

    # ------------------------------------------------------------------------
    # Start very gently.
    # ------------------------------------------------------------------------

    x = (
        L -
        cfg.highlight_start
    ) / (
        cfg.highlight_roll_end -
        cfg.highlight_start
    )

    mask = _smoothstep(
        x
    )

    # Stronger only when actual clipping exists.
    strength = cfg.highlight_strength
    rec = _get_recommendations(analysis)
    if "highlights" in rec:
        # v3 recommendation is negative for highlight recovery.
        strength = _clamp(abs(_safe_float(rec.get("highlights", 0.0))) / 100.0 * 1.6, 0.0, 0.65)

    if clipping > 1.0:
        strength *= 1.35
    elif clipping > 0.30:
        strength *= 1.15

    if bright_pct > 12.0:
        strength *= 1.25

    strength = _clamp(
        strength,
        0.0,
        0.65,
    )

    # Soft compression toward the upper tonal range. This works on local
    # overexposed areas without pulling the whole image into darkness.
    excess = np.maximum(
        L - cfg.highlight_roll_start,
        0.0,
    )

    compression = (
        excess *
        strength *
        mask *
        0.28
    )

    L -= compression
    L = np.maximum(L, 0.0)

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# SHADOW OPENING
# ============================================================================

def _apply_shadow_opening(
    img: np.ndarray,
    analysis: dict[str, Any],
    exposure: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    clipping = _safe_float(
        exposure.get(
            "shadow_clipping",
            {},
        ).get(
            "all_channels_percentage",
            0.0,
        )
    )

    dark_pct = _safe_float(
        exposure.get(
            "dark_pixels_percentage",
            0.0,
        )
    )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    L = lab[:, :, 0].astype(
        np.float32
    )

    # Lift actual shadow regions even when the image as a whole is not dark.
    if not np.any(L < 65.0):
        return img

    # ------------------------------------------------------------------------
    # Only affect deep shadows.
    # ------------------------------------------------------------------------

    x = (
        cfg.shadow_open_end -
        L
    ) / (
        cfg.shadow_open_end -
        cfg.shadow_start
    )

    mask = _smoothstep(
        x
    )

    strength = cfg.shadow_strength
    rec = _get_recommendations(analysis)
    if "shadows" in rec:
        strength = _clamp(_safe_float(rec.get("shadows", 0.0)) / 100.0 * 1.8, 0.0, 0.55)

    if clipping > 2.0:
        strength *= 0.65
    elif clipping > 0.5:
        strength *= 0.85

    if dark_pct > 18.0:
        strength *= 1.15

    # ------------------------------------------------------------------------
    # Nonlinear lift.
    #
    # Extremely black pixels cannot magically be recovered.
    # ------------------------------------------------------------------------

    lift = (
        7.0 *
        strength *
        mask
    )

    # Reduce lift at absolute black.
    black_protection = np.clip(
        L / 18.0,
        0.0,
        1.0,
    )

    lift *= (
        0.30 +
        0.70 *
        black_protection
    )

    L += lift

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# LOCAL CONTRAST / CLARITY
# ============================================================================

def _calculate_clarity_amount(
    analysis: dict[str, Any],
    cfg: EnhancementConfig,
) -> float:

    quality = _get_quality(
        analysis
    )

    noise = _get_noise(
        quality
    )

    contrast = analysis.get(
        "contrast",
        {},
    )

    if not isinstance(contrast, dict):
        contrast = {}

    local_contrast = _safe_float(
        contrast.get(
            "local_contrast",
            20.0,
        ),
        20.0,
    )

    amount = cfg.clarity_amount

    if local_contrast < 8:
        amount *= 1.35

    elif local_contrast < 14:
        amount *= 1.15

    elif local_contrast > 35:
        amount *= 0.50

    elif local_contrast > 28:
        amount *= 0.70

    if noise > cfg.noise_low:
        amount *= 0.85

    if noise > cfg.noise_medium:
        amount *= 0.65

    if noise > cfg.noise_high:
        amount *= 0.40

    return _clamp(
        amount,
        0.0,
        cfg.clarity_max_amount,
    )


def _apply_clarity(
    img: np.ndarray,
    analysis: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    amount = _calculate_clarity_amount(
        analysis,
        cfg,
    )

    if amount < 0.01:
        return img

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    L = lab[:, :, 0]

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        cfg.clarity_sigma,
    )

    local_detail = (
        L.astype(np.float32) -
        blur.astype(np.float32)
    )

    # ------------------------------------------------------------------------
    # Prevent extreme local contrast.
    # ------------------------------------------------------------------------

    local_detail = np.clip(
        local_detail,
        -30.0,
        30.0,
    )

    enhanced = (
        L.astype(np.float32) +
        local_detail *
        amount
    )

    lab[:, :, 0] = np.clip(
        enhanced,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# VIBRANCE
# ============================================================================

def _build_vibrance_lut(
    boost: float,
) -> np.ndarray:

    s = np.arange(
        256,
        dtype=np.float32,
    )

    # Protect high saturation progressively.
    protection = (
        1.0 -
        np.power(
            s / 255.0,
            1.65,
        )
    )

    factor = (
        1.0 +
        (boost - 1.0) *
        protection
    )

    return np.clip(
        s * factor,
        0,
        255,
    ).astype(np.uint8)


def _calculate_vibrance_boost(
    color: dict[str, Any],
    cfg: EnhancementConfig,
) -> float:

    mean_sat = _get_saturation(
        color
    )

    percentiles = color.get(
        "saturation_percentiles",
        {},
    )

    if not isinstance(percentiles, dict):
        percentiles = {}

    p75 = _safe_float(
        percentiles.get(
            "p75_0",
            mean_sat,
        ),
        mean_sat,
    )

    p90 = _safe_float(
        percentiles.get(
            "p90_0",
            mean_sat,
        ),
        mean_sat,
    )

    high_sat_pct = _safe_float(
        color.get(
            "high_saturation_percentage",
            0.0,
        )
    )

    extreme_sat_pct = _safe_float(
        color.get(
            "extreme_saturation_percentage",
            0.0,
        )
    )

    if mean_sat < 40:
        boost = cfg.vibrance_low_sat

    elif mean_sat < 70:
        boost = (
            1.0 +
            (
                cfg.vibrance_low_sat -
                1.0
            ) *
            0.65
        )

    elif mean_sat < 105:
        boost = cfg.vibrance_medium_sat

    elif mean_sat < 135:
        boost = cfg.vibrance_high_sat

    else:
        boost = cfg.vibrance_already_high

    if p75 > 150:
        boost *= 0.85

    if p90 > 190:
        boost *= 0.70

    if high_sat_pct > 18:
        boost *= 0.78

    if extreme_sat_pct > 5:
        boost *= 0.70

    return max(
        1.0,
        boost,
    )


def _apply_vibrance(
    img: np.ndarray,
    analysis: dict[str, Any],
    color: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    boost = _calculate_vibrance_boost(
        color,
        cfg,
    )

    scene = _get_scene(analysis)
    if str(scene.get("type", "")) == "portrait":
        boost = 1.0 + (boost - 1.0) * 0.72

    if boost <= 1.002:
        return img

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV,
    )

    S = hsv[:, :, 1]

    lut = _build_vibrance_lut(
        boost
    )

    S = cv2.LUT(
        S,
        lut,
    )

    S = np.minimum(
        S,
        cfg.saturation_limit,
    ).astype(np.uint8)

    hsv[:, :, 1] = S

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )


# ============================================================================
# SHARPENING
# ============================================================================

def _calculate_sharpen_amount(
    quality: dict[str, Any],
    cfg: EnhancementConfig,
) -> float:

    sharpness = _get_sharpness(
        quality
    )

    noise = _get_noise(
        quality
    )

    # ------------------------------------------------------------------------
    # Good/high detail = don't sharpen.
    # ------------------------------------------------------------------------

    if sharpness >= cfg.sharpness_high_threshold:
        return 0.0

    if sharpness >= cfg.sharpness_good_threshold:
        amount = 0.05

    elif sharpness >= cfg.sharpness_soft_threshold:
        amount = cfg.sharp_amount * 0.60

    else:
        amount = cfg.sharp_amount

    # Very soft images can use slightly more.
    if sharpness < 45:
        amount *= 1.15

    # ------------------------------------------------------------------------
    # Noise protection.
    # ------------------------------------------------------------------------

    if noise > cfg.noise_low:
        amount *= 0.82

    if noise > cfg.noise_medium:
        amount *= 0.62

    if noise > cfg.noise_high:
        amount *= 0.35

    return _clamp(
        amount,
        0.0,
        cfg.sharp_max_amount,
    )


def _apply_sharpening(
    img: np.ndarray,
    quality: dict[str, Any],
    cfg: EnhancementConfig,
) -> np.ndarray:

    amount = _calculate_sharpen_amount(
        quality,
        cfg,
    )

    if amount < 0.01:
        return img

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB,
    )

    L = lab[:, :, 0].astype(
        np.float32
    )

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        cfg.sharp_sigma,
    )

    detail = (
        L -
        blur
    )

    # Prevent sharpening halos.
    detail = np.clip(
        detail,
        -22.0,
        22.0,
    )

    enhanced = (
        L +
        detail *
        amount
    )

    lab[:, :, 0] = np.clip(
        enhanced,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# FINAL COLOR SAFETY
# ============================================================================

def _restore_midtone_luminance(
    original: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    """Lift the image slightly when neutral WB is correct but the tone went too dark."""

    original_luma = _calculate_luma(original).mean()
    current_luma = _calculate_luma(img).mean()

    if current_luma >= original_luma * 0.985:
        return img

    lift = (original_luma - current_luma) * 0.42
    if lift <= 0.5:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    L = np.clip(L + lift, 0, 255)
    lab[:, :, 0] = L.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _final_color_safety(
    img: np.ndarray,
) -> np.ndarray:

    return np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# GLOBAL CHANGE LIMITER
# ============================================================================

def _limit_extreme_global_change(
    original: np.ndarray,
    enhanced: np.ndarray,
    cfg: EnhancementConfig,
) -> np.ndarray:
    """
    Prevent catastrophic global changes.

    This is a safety net, not the primary enhancement mechanism.
    """

    original_luma = _calculate_luma(
        original
    )

    enhanced_luma = _calculate_luma(
        enhanced
    )

    difference = (
        enhanced_luma -
        original_luma
    )

    # Only clamp extreme deviations.
    difference = np.clip(
        difference,
        -cfg.max_total_luma_change,
        cfg.max_total_luma_change,
    )

    corrected_luma = (
        original_luma +
        difference
    )

    # Calculate correction ratio.
    ratio = (
        corrected_luma /
        np.maximum(
            enhanced_luma,
            1.0,
        )
    )

    output = enhanced.astype(
        np.float32
    )

    output *= ratio[:, :, None]

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# NATURAL STRENGTH
# ============================================================================

def _calculate_effective_strength(
    strength: float,
) -> float:
    """
    Convert user strength into a safer blend amount.

    >1 does not extrapolate indefinitely.
    """

    strength = _clamp(
        strength,
        0.0,
        2.0,
    )

    if strength <= 1.0:
        return strength

    # Compress 1..2 into approximately 1..1.30.
    extra = strength - 1.0

    return (
        1.0 +
        extra *
        0.30
    )


# ============================================================================
# PUBLIC API
# ============================================================================

def enhance(
    image: np.ndarray,
    analysis: dict[str, Any],
    cfg: EnhancementConfig | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Natural adaptive photographic enhancement.

    Parameters
    ----------
    image:
        Input BGR uint8 image.

    analysis:
        Analysis JSON dictionary.

    cfg:
        Enhancement configuration.

    strength:
        0.0 = original
        0.5 = subtle
        1.0 = normal
        1.5 = stronger
        2.0 = maximum controlled strength
    """

    if cfg is None:
        cfg = EnhancementConfig()

    # ------------------------------------------------------------------------
    # Validate image.
    # ------------------------------------------------------------------------

    if image is None:
        raise EnhancementError(
            "Input image is None."
        )

    if image.ndim != 3:
        raise EnhancementError(
            f"Expected 3D image, got {image.ndim}D."
        )

    if image.shape[2] != 3:
        raise EnhancementError(
            (
                "Expected 3-channel BGR image, "
                f"got {image.shape[2]} channels."
            )
        )

    if image.dtype != np.uint8:
        raise EnhancementError(
            f"Expected uint8 image, got {image.dtype}."
        )

    if not isinstance(
        analysis,
        dict,
    ):
        raise EnhancementError(
            "Analysis must be a dictionary."
        )

    # ------------------------------------------------------------------------
    # Analysis sections.
    # ------------------------------------------------------------------------

    exposure = _get_exposure(
        analysis
    )

    color = _get_color(
        analysis
    )

    quality = _get_quality(
        analysis
    )

    original = image.copy()

    out = image.copy()

    # ------------------------------------------------------------------------
    # 1. WHITE BALANCE
    # ------------------------------------------------------------------------

    if cfg.enable_wb:

        out = _apply_white_balance(
            out,
            analysis,
            color,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 2. EXPOSURE
    # ------------------------------------------------------------------------

    if cfg.enable_tone:

        out = _apply_exposure(
            out,
            analysis,
            exposure,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 3. NATURAL CONTRAST
    # ------------------------------------------------------------------------

    if cfg.enable_tone:

        out = _apply_natural_contrast(
            out,
            analysis,
            exposure,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 4. HIGHLIGHT ROLLOFF
    # ------------------------------------------------------------------------

    if cfg.enable_highlight_protection:

        out = _apply_highlight_rolloff(
            out,
            analysis,
            exposure,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 5. SHADOW OPENING
    # ------------------------------------------------------------------------

    if cfg.enable_shadow_protection:

        out = _apply_shadow_opening(
            out,
            analysis,
            exposure,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 6. CLARITY
    # ------------------------------------------------------------------------

    if cfg.enable_clarity:

        out = _apply_clarity(
            out,
            analysis,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 7. VIBRANCE
    # ------------------------------------------------------------------------

    if cfg.enable_vibrance:

        out = _apply_vibrance(
            out,
            analysis,
            color,
            cfg,
        )

    # ------------------------------------------------------------------------
    # 8. SHARPEN
    # ------------------------------------------------------------------------

    if cfg.enable_sharpen:

        out = _apply_sharpening(
            out,
            quality,
            cfg,
        )

    # ------------------------------------------------------------------------
    # Final neutral verification pass.
    # This corrects only small residual WB errors measured on the rendered image.
    # ------------------------------------------------------------------------

    if cfg.enable_wb:
        out = _apply_residual_white_balance(out, cfg)

    # ------------------------------------------------------------------------
    # Safety.
    # ------------------------------------------------------------------------

    out = _final_color_safety(
        out
    )

    out = _limit_extreme_global_change(
        original,
        out,
        cfg,
    )

    # ------------------------------------------------------------------------
    # Global strength.
    # ------------------------------------------------------------------------

    effective_strength = (
        _calculate_effective_strength(
            strength
        )
    )

    if effective_strength <= 0.001:
        return original

    if abs(
        effective_strength - 1.0
    ) > 0.001:

        out = cv2.addWeighted(
            original,
            1.0 - effective_strength,
            out,
            effective_strength,
            0.0,
        )

        out = np.clip(
            out,
            0,
            255,
        ).astype(np.uint8)

    return out


# ============================================================================
# JSON LOADING
# ============================================================================

def _load_analysis(
    path: Path,
) -> dict[str, Any]:

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

    except json.JSONDecodeError as exc:

        raise EnhancementError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    except OSError as exc:

        raise EnhancementError(
            f"Cannot read {path}: {exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise EnhancementError(
            "Analysis JSON root must be an object."
        )

    # Keep compatibility with your existing analyzer.
    required = (
        "exposure_and_tone",
        "color_profile",
        "quality",
    )

    for key in required:

        if key not in data:

            raise EnhancementError(
                f"Analysis JSON is missing '{key}'."
            )

    return data


# ============================================================================
# IMAGE SAVING
# ============================================================================

def _save_image(
    img: np.ndarray,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ext = path.suffix.lower()

    params: list[int] = []

    if ext in (
        ".jpg",
        ".jpeg",
    ):

        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            97,
        ]

    elif ext == ".png":

        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    elif ext in (
        ".tif",
        ".tiff",
    ):

        params = [
            cv2.IMWRITE_TIFF_COMPRESSION,
            1,
        ]

    success = cv2.imwrite(
        str(path),
        img,
        params,
    )

    if not success:

        raise EnhancementError(
            f"Failed to write image: {path}"
        )


# ============================================================================
# SUMMARY
# ============================================================================

def _print_analysis_summary(
    analysis: dict[str, Any],
) -> None:

    exposure = _get_exposure(
        analysis
    )

    color = _get_color(
        analysis
    )

    quality = _get_quality(
        analysis
    )

    wb = color.get(
        "white_balance",
        {},
    )

    if not isinstance(wb, dict):
        wb = {}

    clipping_high = exposure.get(
        "highlight_clipping",
        {},
    )

    clipping_shadow = exposure.get(
        "shadow_clipping",
        {},
    )

    if not isinstance(
        clipping_high,
        dict,
    ):
        clipping_high = {}

    if not isinstance(
        clipping_shadow,
        dict,
    ):
        clipping_shadow = {}

    print()
    print("-" * 72)
    print("IMAGE ANALYSIS")
    print("-" * 72)

    print(
        f"Mean luminance      : "
        f"{_get_mean_luminance(exposure):.2f}"
    )

    print(
        f"Median luminance    : "
        f"{_get_median_luminance(exposure):.2f}"
    )

    print(
        f"Dynamic range       : "
        f"{_safe_float(exposure.get('effective_dynamic_range', 0)):.2f}"
    )

    print(
        f"Dark pixels         : "
        f"{_safe_float(exposure.get('dark_pixels_percentage', 0)):.2f}%"
    )

    print(
        f"Bright pixels       : "
        f"{_safe_float(exposure.get('bright_pixels_percentage', 0)):.2f}%"
    )

    print(
        f"Highlight clipping  : "
        f"{_safe_float(clipping_high.get('all_channels_percentage', 0)):.4f}%"
    )

    print(
        f"Shadow clipping     : "
        f"{_safe_float(clipping_shadow.get('all_channels_percentage', 0)):.4f}%"
    )

    print(
        f"Saturation          : "
        f"{_get_saturation(color):.2f}"
    )

    print(
        f"WB temperature cast : "
        f"{_safe_float(wb.get('estimated_temperature_cast', 0)):.3f}"
    )

    print(
        f"WB tint cast        : "
        f"{_safe_float(wb.get('estimated_tint_cast', 0)):.3f}"
    )

    print(
        f"WB neutral coverage : "
        f"{_safe_float(wb.get('neutral_pixel_coverage_percentage', 0)):.2f}%"
    )

    print(
        f"Sharpness           : "
        f"{_get_sharpness(quality):.3f}"
    )

    print(
        f"Noise               : "
        f"{_get_noise(quality):.3f}"
    )

    rec = _get_recommendations(analysis)
    if rec:
        ev = rec.get("exposure_ev", {})
        ev_value = _safe_float(ev.get("value", 0.0)) if isinstance(ev, dict) else _safe_float(ev)
        print(f"Recommended EV      : {ev_value:+.3f}")
        print(f"Recommended WB      : {rec.get('temperature', {}).get('value', 0.0) if isinstance(rec.get('temperature'), dict) else 0.0:+.2f} / {rec.get('tint', {}).get('value', 0.0) if isinstance(rec.get('tint'), dict) else 0.0:+.2f}")
        print(f"Scene               : {_get_scene(analysis).get('type', 'general')}")

    print("-" * 72)
    print()


# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Natural adaptive photographic enhancement "
            "using image-analysis JSON."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "input_image",
        type=Path,
        help="Input image.",
    )

    parser.add_argument(
        "analysis_json",
        type=Path,
        help="Analysis JSON.",
    )

    parser.add_argument(
        "output_image",
        type=Path,
        help="Output image.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Overall enhancement strength. "
            "0=original, 1=normal, 2=maximum."
        ),
    )

    parser.add_argument(
        "--no-wb",
        action="store_true",
        help="Disable white balance.",
    )

    parser.add_argument(
        "--no-tone",
        action="store_true",
        help="Disable exposure and contrast.",
    )

    parser.add_argument(
        "--no-protection",
        action="store_true",
        help=(
            "Disable highlight and shadow protection."
        ),
    )

    parser.add_argument(
        "--no-clarity",
        action="store_true",
        help="Disable clarity.",
    )

    parser.add_argument(
        "--no-vibrance",
        action="store_true",
        help="Disable vibrance.",
    )

    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable sharpening.",
    )

    parser.add_argument(
        "--clarity",
        type=float,
        default=None,
        help="Override clarity amount.",
    )

    parser.add_argument(
        "--wb-strength",
        type=float,
        default=None,
        help="Override white-balance strength.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main(
    argv: list[str] | None = None,
) -> int:

    parser = _build_parser()

    args = parser.parse_args(
        argv
    )

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------------
    # Validate files.
    # ------------------------------------------------------------------------

    if not args.input_image.is_file():

        logger.error(
            "Input image not found: %s",
            args.input_image,
        )

        return 2

    if not args.analysis_json.is_file():

        logger.error(
            "Analysis JSON not found: %s",
            args.analysis_json,
        )

        return 2

    # ------------------------------------------------------------------------
    # Load image.
    # ------------------------------------------------------------------------

    logger.info(
        "Loading image: %s",
        args.input_image,
    )

    try:

        image = decode_color(
            args.input_image
        )

    except Exception as exc:

        logger.error(
            "Failed to load image: %s",
            exc,
        )

        return 1

    # ------------------------------------------------------------------------
    # Load analysis.
    # ------------------------------------------------------------------------

    logger.info(
        "Loading analysis: %s",
        args.analysis_json,
    )

    try:

        analysis = _load_analysis(
            args.analysis_json
        )

    except EnhancementError as exc:

        logger.error(
            "%s",
            exc,
        )

        return 1

    # ------------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------------

    _print_analysis_summary(
        analysis
    )

    # ------------------------------------------------------------------------
    # Configuration.
    # ------------------------------------------------------------------------

    default_cfg = EnhancementConfig()

    clarity_amount = (
        args.clarity
        if args.clarity is not None
        else default_cfg.clarity_amount
    )

    wb_strength = (
        args.wb_strength
        if args.wb_strength is not None
        else default_cfg.wb_strength
    )

    cfg = EnhancementConfig(
        wb_strength=wb_strength,
        clarity_amount=clarity_amount,

        enable_wb=not args.no_wb,

        enable_tone=not args.no_tone,

        enable_highlight_protection=(
            not args.no_protection
        ),

        enable_shadow_protection=(
            not args.no_protection
        ),

        enable_clarity=not args.no_clarity,

        enable_vibrance=not args.no_vibrance,

        enable_sharpen=not args.no_sharpen,
    )

    # ------------------------------------------------------------------------
    # Enhance.
    # ------------------------------------------------------------------------

    logger.info(
        "Running natural adaptive enhancement "
        "(strength=%.2f)...",
        args.strength,
    )

    try:

        enhanced = enhance(
            image,
            analysis,
            cfg=cfg,
            strength=args.strength,
        )

    except EnhancementError as exc:

        logger.error(
            "Enhancement failed: %s",
            exc,
        )

        return 1

    except Exception as exc:

        logger.exception(
            "Unexpected enhancement failure: %s",
            exc,
        )

        return 1

    # ------------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------------

    try:

        _save_image(
            enhanced,
            args.output_image,
        )

    except EnhancementError as exc:

        logger.error(
            "%s",
            exc,
        )

        return 1

    logger.info(
        "Saved enhanced image: %s",
        args.output_image,
    )

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )