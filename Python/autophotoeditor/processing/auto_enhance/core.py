from __future__ import annotations

"""
AutoPhotoEditor - Enhancement Engine V4
=======================================

A conservative, photographic enhancement engine that consumes the canonical
analysis JSON produced by the AutoPhotoEditor analysis pipeline.

Key changes from the previous version
--------------------------------------
1. White balance and exposure are performed in linear RGB.
2. There is one authoritative tone pipeline; no competing library tone pass.
3. White-balance gains are luminance-normalized.
4. Highlights use soft roll-off instead of global luminance scaling.
5. Gamut protection reduces chroma before clipping.
6. Every edit is confidence-weighted and bounded by an edit budget.
7. Enhancement can validate the result against the source.
8. EnhancementConfig switches are authoritative.
9. strength is consistently limited to [0, 1].
10. The module remains compatible with the canonical analysis/correction_plan
    JSON structure used by AutoPhotoEditor.

Input
-----
BGR uint8 NumPy image.

Output
------
BGR uint8 NumPy image.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("autophotoeditor.enhancement_v4")

EPS = 1e-8
SCHEMA_VERSION = "4.0"
DEFAULT_ANALYSIS_MAX_DIMENSION = 1800
DEFAULT_DETAIL_MAX_DIMENSION = 1600


# ============================================================
# ERRORS / CONFIGURATION
# ============================================================


class EnhancementError(Exception):
    """Raised when an enhancement operation cannot be completed safely."""


@dataclass(frozen=True)
class EditBudget:
    """Hard safety limits for automatic photographic correction."""

    max_exposure_ev: float = 1.50
    max_wb_deviation: float = 0.35
    max_highlight_recovery: float = 0.40
    max_shadow_recovery: float = 0.40
    max_contrast: float = 0.25
    max_whites: float = 0.25
    max_blacks: float = 0.25
    max_saturation_change: float = 0.12
    max_color_error: float = 0.10
    max_luma_ratio: float = 2.20
    min_luma_ratio: float = 0.35


@dataclass(frozen=True)
class EnhancementConfig:
    """Controls the enhancement pipeline.

    strength is supplied to enhance(); all stage switches below are honored.
    """

    wb_strength: float = 1.0
    tone_strength: float = 1.0
    clarity_amount: float = 0.0

    enable_wb: bool = True
    enable_exposure: bool = True
    enable_highlight_protection: bool = True
    enable_shadow_protection: bool = True
    enable_contrast: bool = True
    enable_whites_blacks: bool = True
    enable_color_tone: bool = True
    enable_clarity: bool = False
    enable_vibrance: bool = False
    enable_sharpen: bool = False
    enable_gamut_protection: bool = True
    enable_validation: bool = True

    highlight_rolloff_start: float = 0.72
    highlight_rolloff_end: float = 0.995
    shadow_recovery_start: float = 0.24
    shadow_recovery_end: float = 0.52

    budget: EditBudget = EditBudget()


@dataclass(frozen=True)
class Estimate:
    value: float
    confidence: float

    def to_json(self, digits: int = 4) -> Dict[str, float]:
        return {
            "value": round(float(self.value), digits),
            "confidence": round(float(np.clip(self.confidence, 0.0, 1.0)), digits),
        }


# ============================================================
# GENERIC HELPERS
# ============================================================


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _number(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default)


def clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def sigmoid(x: float) -> float:
    x = clamp(x, -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-x))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size != w.size:
        raise ValueError("values and weights must have the same length")
    s = float(np.sum(w))
    if s <= EPS:
        return float(np.mean(v))
    return float(np.sum(v * w) / s)


def percentile_dict(channel: np.ndarray, percentiles: Sequence[float]) -> Dict[str, float]:
    values = np.percentile(channel, tuple(percentiles))
    result: Dict[str, float] = {}
    for p, value in zip(percentiles, values):
        key = f"p{str(p).replace('.', '_')}"
        result[key] = round(safe_float(value), 4)
    return result


def normalized_histogram(
    channel: np.ndarray,
    bins: int = 256,
    value_range: Tuple[float, float] = (0, 256),
) -> list[float]:
    arr = np.asarray(channel)
    if arr.dtype == np.uint8 and value_range == (0, 256):
        hist = cv2.calcHist([arr], [0], None, [bins], list(value_range)).flatten()
    else:
        hist, _ = np.histogram(arr, bins=bins, range=value_range)
        hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    return [round(float(x), 8) for x in hist]


def resize_longest(image: np.ndarray, max_dimension: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.bool_:
        m = m > 0.5 if np.issubdtype(m.dtype, np.floating) else m > 0
    return m.astype(bool)


def mask_coverage(mask: np.ndarray) -> float:
    return float(np.mean(mask) * 100.0) if mask.size else 0.0


def robust_median(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return safe_float(np.median(values), default)


def robust_mean(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return safe_float(np.mean(values), default)


def robust_mad(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return safe_float(mad, default)


# ============================================================
# COLOR / LINEAR-LIGHT HELPERS
# ============================================================


def validate_bgr_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise EnhancementError("Expected a NumPy image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise EnhancementError("Expected a 3-channel BGR image.")
    if image.size == 0:
        raise EnhancementError("Image is empty.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def bgr8_to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def rgb_float_to_bgr8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.clip(np.round(bgr * 255.0), 0, 255).astype(np.uint8)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    rgb_linear = np.clip(rgb_linear.astype(np.float32), 0.0, 1.0)
    return np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def linear_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    ).astype(np.float32)


def srgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return linear_luminance(srgb_to_linear(rgb))


def luminance_to_ev(y: np.ndarray, reference: float = 0.18) -> np.ndarray:
    return np.log2(np.maximum(y, 1e-6) / reference)


def chromaticity_rg(rgb_linear: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = np.sum(rgb_linear, axis=2) + EPS
    r = rgb_linear[:, :, 0] / total
    g = rgb_linear[:, :, 1] / total
    return r, g


def luma_percentiles(rgb_linear: np.ndarray) -> Dict[str, float]:
    y = linear_luminance(rgb_linear)
    p = np.percentile(y, [0.5, 1, 5, 18, 50, 82, 95, 99, 99.5])
    return {
        "p0_5": float(p[0]),
        "p1": float(p[1]),
        "p5": float(p[2]),
        "p18": float(p[3]),
        "p50": float(p[4]),
        "p82": float(p[5]),
        "p95": float(p[6]),
        "p99": float(p[7]),
        "p99_5": float(p[8]),
    }


# ============================================================
# ANALYSIS PLAN EXTRACTION
# ============================================================


def _get_edit_plan(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = _as_mapping(analysis.get("correction_plan"))
    if plan:
        return plan
    return _as_mapping(analysis.get("recommendations"))


def _section_value(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if isinstance(value, Mapping):
                if "value" in value:
                    return value["value"]
                if "recommended_change_ev" in value:
                    return value["recommended_change_ev"]
            return value
    return default


def _get_recommendation_value(
    analysis: Mapping[str, Any],
    section: str,
    default: float = 0.0,
) -> float:
    plan = _get_edit_plan(analysis)
    value = plan.get(section, default)

    if section not in plan:
        light = _as_mapping(plan.get("light"))
        value = light.get(section, default)

    if isinstance(value, Mapping):
        value = value.get("value", value.get("recommended_change_ev", default))

    if value == default:
        best = _as_mapping(analysis.get("best_enhancement"))
        luminosity = _as_mapping(best.get("luminosity"))
        neutral = _as_mapping(best.get("neutral_tone"))
        if section == "exposure_ev":
            value = luminosity.get("recommended_change_ev", default)
        elif section in {"highlights", "shadows", "contrast", "whites", "blacks"}:
            value = luminosity.get(section, default)
        elif section == "color_tone":
            value = neutral.get("value", default)

    return _number(value, default)


def _get_confidence(
    analysis: Mapping[str, Any],
    section: str,
    default: float = 0.0,
) -> float:
    plan = _get_edit_plan(analysis)
    raw = plan.get(section)
    if isinstance(raw, Mapping):
        return clamp(_number(raw.get("confidence"), default), 0.0, 1.0)

    best = _as_mapping(analysis.get("best_enhancement"))
    section_map = _as_mapping(best.get(section))
    if section_map:
        return clamp(_number(section_map.get("confidence"), default), 0.0, 1.0)

    return clamp(default, 0.0, 1.0)


def _get_wb_gains(analysis: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    white_balance = _as_mapping(analysis.get("white_balance"))
    plan = _get_edit_plan(analysis)
    plan_wb = _as_mapping(plan.get("white_balance"))

    raw = plan_wb.get(
        "gains_rgb",
        white_balance.get(
            "gains_rgb",
            white_balance.get("correction_gains_rgb", []),
        ),
    )

    try:
        gains = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        gains = np.empty(0, dtype=np.float32)

    if gains.size != 3 or not np.all(np.isfinite(gains)):
        return np.empty(0, dtype=np.float32), 0.0

    confidence = _number(
        plan_wb.get(
            "confidence",
            white_balance.get(
                "confidence",
                _as_mapping(white_balance.get("estimated_temperature_cast")).get(
                    "confidence", 0.0
                ),
            ),
        ),
        0.0,
    )
    return gains, clamp(confidence, 0.0, 1.0)


# ============================================================
# WHITE BALANCE
# ============================================================


def normalize_wb_gains(gains: np.ndarray) -> np.ndarray:
    """Normalize WB so it primarily changes chroma rather than exposure."""
    gains = np.asarray(gains, dtype=np.float32).reshape(3)
    gains = np.clip(gains, 0.65, 1.50)

    # Preserve green as the reference while removing the average luminance
    # effect of the correction.
    weighted = float(np.dot(gains, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    if weighted <= EPS:
        return np.ones(3, dtype=np.float32)
    return np.clip(gains / weighted, 0.65, 1.50).astype(np.float32)


def apply_white_balance_linear(
    rgb_linear: np.ndarray,
    gains: np.ndarray,
    strength: float,
    confidence: float,
    max_deviation: float,
) -> np.ndarray:
    if gains.size != 3:
        return rgb_linear

    normalized = normalize_wb_gains(gains)
    delta = normalized - 1.0
    delta = np.clip(delta, -max_deviation, max_deviation)
    factor = clamp(strength * confidence, 0.0, 1.0)
    effective = 1.0 + delta * factor

    return np.clip(rgb_linear * effective.reshape(1, 1, 3), 0.0, None).astype(np.float32)


# ============================================================
# EXPOSURE / TONE
# ============================================================


def apply_exposure_linear(
    rgb_linear: np.ndarray,
    exposure_ev: float,
    strength: float,
    confidence: float,
    budget: EditBudget,
) -> np.ndarray:
    ev = clamp(exposure_ev, -budget.max_exposure_ev, budget.max_exposure_ev)
    amount = ev * clamp(strength * confidence, 0.0, 1.0)
    if abs(amount) < 1e-5:
        return rgb_linear
    return np.clip(rgb_linear * np.float32(2.0 ** amount), 0.0, None).astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_highlight_protection(
    rgb_linear: np.ndarray,
    amount: float,
    strength: float,
    confidence: float,
    start: float,
    end: float,
    budget: EditBudget,
) -> np.ndarray:
    if amount >= 0.0:
        return rgb_linear

    y = linear_luminance(rgb_linear)
    protection = min(abs(amount) / 100.0, budget.max_highlight_recovery)
    protection *= clamp(strength * confidence, 0.0, 1.0)
    if protection <= 1e-5:
        return rgb_linear

    weight = _smoothstep(start, end, y)
    # Soft compression.  It approaches 1-protection at the very top instead
    # of suddenly scaling the whole image.
    scale = 1.0 - protection * weight
    return np.clip(rgb_linear * scale[:, :, None], 0.0, None).astype(np.float32)


def apply_shadow_recovery(
    rgb_linear: np.ndarray,
    amount: float,
    strength: float,
    confidence: float,
    start: float,
    end: float,
    budget: EditBudget,
) -> np.ndarray:
    if amount <= 0.0:
        return rgb_linear

    y = linear_luminance(rgb_linear)
    recovery = min(amount / 100.0, budget.max_shadow_recovery)
    recovery *= clamp(strength * confidence, 0.0, 1.0)
    if recovery <= 1e-5:
        return rgb_linear

    weight = 1.0 - _smoothstep(start, end, y)
    # Keep deep blacks from becoming gray.  The multiplier falls to zero near
    # absolute black and reaches its maximum in recoverable shadow regions.
    weight *= np.clip(y / max(end, EPS), 0.0, 1.0) ** 0.35
    lift = recovery * weight
    result = rgb_linear + lift[:, :, None] * 0.12
    return np.clip(result, 0.0, None).astype(np.float32)


def apply_contrast_linear(
    rgb_linear: np.ndarray,
    contrast: float,
    strength: float,
    confidence: float,
    budget: EditBudget,
) -> np.ndarray:
    c = clamp(contrast / 100.0, -budget.max_contrast, budget.max_contrast)
    factor = 1.0 + c * clamp(strength * confidence, 0.0, 1.0)
    if abs(factor - 1.0) < 1e-5:
        return rgb_linear

    y = linear_luminance(rgb_linear)
    # 18% gray is a photographic exposure reference and is a more useful
    # pivot for linear-light contrast than 0.5.
    pivot = 0.18
    new_y = np.clip((y - pivot) * factor + pivot, 0.0, None)
    ratio = new_y / np.maximum(y, 1e-6)
    return np.clip(rgb_linear * ratio[:, :, None], 0.0, None).astype(np.float32)


def apply_whites_blacks(
    rgb_linear: np.ndarray,
    whites: float,
    blacks: float,
    strength: float,
    confidence: float,
    budget: EditBudget,
) -> np.ndarray:
    w = clamp(whites / 100.0, -budget.max_whites, budget.max_whites)
    b = clamp(blacks / 100.0, -budget.max_blacks, budget.max_blacks)
    factor = clamp(strength * confidence, 0.0, 1.0)
    w *= factor
    b *= factor

    if abs(w) < 1e-5 and abs(b) < 1e-5:
        return rgb_linear

    y = linear_luminance(rgb_linear)
    high = _smoothstep(0.55, 0.95, y)
    low = 1.0 - _smoothstep(0.02, 0.40, y)
    delta = w * high + b * low
    new_y = np.clip(y + delta, 0.0, None)
    ratio = new_y / np.maximum(y, 1e-6)
    return np.clip(rgb_linear * ratio[:, :, None], 0.0, None).astype(np.float32)


# ============================================================
# COLOR TONE / DETAIL
# ============================================================


def apply_color_tone_neutralization(
    rgb_linear: np.ndarray,
    analysis: Mapping[str, Any],
    strength: float,
) -> np.ndarray:
    """Optional conservative LAB neutralization using analyzer guidance.

    This is deliberately weak. The authoritative WB correction is gains_rgb.
    The stage exists only for a residual neutral-tone recommendation.
    """
    plan = _get_edit_plan(analysis)
    tone = _as_mapping(plan.get("color_tone"))
    if not tone:
        tone = _as_mapping(plan.get("neutral_tone"))

    value = _number(tone.get("value", 0.0))
    confidence = clamp(_number(tone.get("confidence", 0.0)), 0.0, 1.0)
    if abs(value) < 1e-5 or confidence <= 0.0:
        return rgb_linear

    # Convert a small neutral correction through sRGB/Lab. This is intentionally
    # capped so it cannot override legitimate scene color.
    srgb = linear_to_srgb(rgb_linear)
    lab = cv2.cvtColor(np.clip(srgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    correction = clamp(value / 100.0, -0.08, 0.08)
    factor = clamp(strength * confidence, 0.0, 1.0)
    neutral_weight = np.clip(1.0 - (np.abs(a) + np.abs(b)) / 90.0, 0.0, 1.0)
    lab[:, :, 1] = np.clip(128.0 + a - a * correction * factor * neutral_weight, 0, 255)
    lab[:, :, 2] = np.clip(128.0 + b - b * correction * factor * neutral_weight, 0, 255)

    corrected_srgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(corrected_srgb)


def apply_vibrance(rgb_linear: np.ndarray, amount: float, strength: float) -> np.ndarray:
    if abs(amount) < 1e-5:
        return rgb_linear

    srgb = linear_to_srgb(rgb_linear)
    hsv = cv2.cvtColor(
        np.clip(srgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    ).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    gain = np.clip(amount / 100.0, -0.12, 0.12) * clamp(strength, 0.0, 1.0)
    # Protect already saturated colors and preserve skin-like low-chroma tones.
    factor = gain * (1.0 - saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + factor), 0.0, 255.0)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(result)


def apply_clarity(rgb_linear: np.ndarray, amount: float, strength: float) -> np.ndarray:
    if amount <= 0.0 or strength <= 0.0:
        return rgb_linear

    srgb = linear_to_srgb(rgb_linear)
    lab = cv2.cvtColor(np.clip(srgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=4.0)
    detail = l - blur
    gain = clamp(amount, 0.0, 0.10) * clamp(strength, 0.0, 1.0)
    lab[:, :, 0] = np.clip(l + detail * (gain * 3.0), 0.0, 255.0)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(result)


def apply_sharpen(rgb_linear: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return rgb_linear
    srgb = linear_to_srgb(rgb_linear)
    blur = cv2.GaussianBlur(srgb, (0, 0), sigmaX=1.0)
    sharpened = srgb + (srgb - blur) * clamp(amount, 0.0, 0.35)
    return srgb_to_linear(np.clip(sharpened, 0.0, 1.0))


# ============================================================
# HIGHLIGHT / GAMUT SAFETY
# ============================================================


def soft_clip_highlights(rgb_linear: np.ndarray, shoulder: float = 0.985) -> np.ndarray:
    """Compress values above the shoulder without altering normal exposure."""
    x = np.maximum(rgb_linear, 0.0)
    over = np.maximum(x - shoulder, 0.0)
    compressed = shoulder + over / (1.0 + over * 5.0)
    return np.where(x > shoulder, compressed, x).astype(np.float32)


def protect_gamut(rgb_linear: np.ndarray) -> np.ndarray:
    """Reduce chroma before final clipping.

    This prevents strong WB/tone edits from producing harsh RGB clipping.
    Luminance is approximately preserved by scaling RGB toward neutral gray.
    """
    x = np.maximum(rgb_linear, 0.0).astype(np.float32)
    y = linear_luminance(x)
    max_channel = np.max(x, axis=2)
    min_channel = np.min(x, axis=2)

    over = np.maximum(max_channel - 1.0, 0.0)
    under = np.maximum(-min_channel, 0.0)
    excess = np.maximum(over, under)

    if not np.any(excess > 0.0):
        return x

    neutral = y[:, :, None]
    # More compression for more severe excursions.
    t = np.clip(excess / np.maximum(excess + 0.25, EPS), 0.0, 1.0)[:, :, None]
    protected = x * (1.0 - t) + neutral * t
    return np.clip(protected, 0.0, 1.0).astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================


def _stats_for_validation(rgb_linear: np.ndarray) -> Dict[str, float]:
    y = linear_luminance(rgb_linear)
    return {
        "median_luma": float(np.median(y)),
        "mean_luma": float(np.mean(y)),
        "p01_luma": float(np.percentile(y, 1.0)),
        "p99_luma": float(np.percentile(y, 99.0)),
        "black_clip": float(np.mean(y <= 1e-5)),
        "highlight_clip": float(np.mean(y >= 0.995)),
        "channel_clip": float(np.mean(np.any(rgb_linear >= 0.999, axis=2))),
        "mean_chroma": float(np.mean(np.max(rgb_linear, axis=2) - np.min(rgb_linear, axis=2))),
    }


def validate_result(
    original_linear: np.ndarray,
    enhanced_linear: np.ndarray,
    budget: EditBudget,
) -> Dict[str, Any]:
    before = _stats_for_validation(original_linear)
    after = _stats_for_validation(enhanced_linear)

    before_median = max(before["median_luma"], 1e-6)
    luma_ratio = after["median_luma"] / before_median
    highlight_delta = after["highlight_clip"] - before["highlight_clip"]
    black_delta = after["black_clip"] - before["black_clip"]
    chroma_delta = after["mean_chroma"] - before["mean_chroma"]

    issues: list[str] = []
    if luma_ratio > budget.max_luma_ratio:
        issues.append("median_luminance_increase_too_large")
    if luma_ratio < budget.min_luma_ratio:
        issues.append("median_luminance_decrease_too_large")
    if highlight_delta > 0.08:
        issues.append("highlight_clipping_increased")
    if black_delta > 0.08:
        issues.append("shadow_clipping_increased")
    if chroma_delta > budget.max_saturation_change:
        issues.append("chroma_increase_too_large")

    accepted = not issues
    return {
        "accepted": accepted,
        "issues": issues,
        "before": before,
        "after": after,
        "changes": {
            "median_luma_ratio": float(luma_ratio),
            "highlight_clip_delta": float(highlight_delta),
            "black_clip_delta": float(black_delta),
            "mean_chroma_delta": float(chroma_delta),
        },
    }


# ============================================================
# MAIN ENHANCEMENT PIPELINE
# ============================================================


def _apply_pipeline(
    rgb_linear: np.ndarray,
    analysis: Mapping[str, Any],
    config: EnhancementConfig,
    effective_strength: float,
) -> np.ndarray:
    budget = config.budget
    enhanced = np.clip(rgb_linear, 0.0, None).astype(np.float32)

    # --------------------------------------------------------
    # 1. WHITE BALANCE - LINEAR RGB
    # --------------------------------------------------------
    if config.enable_wb and config.wb_strength > 0.0:
        gains, confidence = _get_wb_gains(analysis)
        if gains.size == 3 and confidence >= 0.25:
            enhanced = apply_white_balance_linear(
                enhanced,
                gains,
                effective_strength * config.wb_strength,
                confidence,
                budget.max_wb_deviation,
            )

    # --------------------------------------------------------
    # 2. EXPOSURE - LINEAR RGB
    # --------------------------------------------------------
    if config.enable_exposure:
        exposure = _get_recommendation_value(analysis, "exposure_ev", 0.0)
        confidence = _get_confidence(analysis, "exposure_ev", 0.75)
        enhanced = apply_exposure_linear(
            enhanced,
            exposure,
            effective_strength,
            confidence,
            budget,
        )

    # --------------------------------------------------------
    # 3. HIGHLIGHTS
    # --------------------------------------------------------
    if config.enable_highlight_protection:
        highlights = _get_recommendation_value(analysis, "highlights", 0.0)
        confidence = _get_confidence(analysis, "highlights", 0.70)
        enhanced = apply_highlight_protection(
            enhanced,
            highlights,
            effective_strength,
            confidence,
            config.highlight_rolloff_start,
            config.highlight_rolloff_end,
            budget,
        )

    # --------------------------------------------------------
    # 4. SHADOWS
    # --------------------------------------------------------
    if config.enable_shadow_protection:
        shadows = _get_recommendation_value(analysis, "shadows", 0.0)
        confidence = _get_confidence(analysis, "shadows", 0.70)
        enhanced = apply_shadow_recovery(
            enhanced,
            shadows,
            effective_strength,
            confidence,
            config.shadow_recovery_start,
            config.shadow_recovery_end,
            budget,
        )

    # --------------------------------------------------------
    # 5. CONTRAST
    # --------------------------------------------------------
    if config.enable_contrast:
        contrast = _get_recommendation_value(analysis, "contrast", 0.0)
        confidence = _get_confidence(analysis, "contrast", 0.70)
        enhanced = apply_contrast_linear(
            enhanced,
            contrast,
            effective_strength * config.tone_strength,
            confidence,
            budget,
        )

    # --------------------------------------------------------
    # 6. WHITES / BLACKS
    # --------------------------------------------------------
    if config.enable_whites_blacks:
        whites = _get_recommendation_value(analysis, "whites", 0.0)
        blacks = _get_recommendation_value(analysis, "blacks", 0.0)
        confidence = max(
            _get_confidence(analysis, "whites", 0.70),
            _get_confidence(analysis, "blacks", 0.70),
        )
        enhanced = apply_whites_blacks(
            enhanced,
            whites,
            blacks,
            effective_strength * config.tone_strength,
            confidence,
            budget,
        )

    # --------------------------------------------------------
    # 7. RESIDUAL COLOR TONE
    # --------------------------------------------------------
    if config.enable_color_tone:
        enhanced = apply_color_tone_neutralization(
            enhanced,
            analysis,
            effective_strength,
        )

    # --------------------------------------------------------
    # 8. OPTIONAL VIBRANCE / CLARITY
    # --------------------------------------------------------
    if config.enable_vibrance:
        vibrance = _get_recommendation_value(analysis, "vibrance", 0.0)
        enhanced = apply_vibrance(enhanced, vibrance, effective_strength)

    if config.enable_clarity:
        enhanced = apply_clarity(
            enhanced,
            config.clarity_amount,
            effective_strength,
        )

    # --------------------------------------------------------
    # 9. HIGHLIGHT SHOULDER + GAMUT SAFETY
    # --------------------------------------------------------
    if config.enable_highlight_protection:
        enhanced = soft_clip_highlights(enhanced, shoulder=0.985)

    if config.enable_gamut_protection:
        enhanced = protect_gamut(enhanced)

    # --------------------------------------------------------
    # 10. OPTIONAL SHARPEN
    # --------------------------------------------------------
    if config.enable_sharpen:
        enhanced = apply_sharpen(enhanced, effective_strength * 0.20)

    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def enhance(
    image: np.ndarray,
    analysis: Dict[str, Any],
    cfg: EnhancementConfig | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Enhance a BGR image using the canonical analysis JSON.

    The operation is deliberately conservative. If validation detects a major
    technical regression, the pipeline retries at reduced strength. If that
    still fails, the original image is returned unchanged.
    """
    original = validate_bgr_image(image)
    if not isinstance(analysis, Mapping):
        raise EnhancementError("Analysis must be a JSON object.")
    if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
        raise EnhancementError("Strength must be between 0 and 2.")

    config = cfg or EnhancementConfig()
    effective_strength = min(float(strength), 1.0)
    if effective_strength <= 0.0:
        return original.copy()

    rgb = bgr8_to_rgb_float(original)
    original_linear = srgb_to_linear(rgb)

    candidate = _apply_pipeline(
        original_linear,
        analysis,
        config,
        effective_strength,
    )

    if not config.enable_validation:
        return rgb_float_to_bgr8(linear_to_srgb(candidate))

    validation = validate_result(
        original_linear,
        candidate,
        config.budget,
    )

    # One conservative retry. This avoids repeatedly compounding edits while
    # still allowing a useful correction when the first pass was too strong.
    if not validation["accepted"] and effective_strength > 0.35:
        retry_strength = effective_strength * 0.55
        candidate_retry = _apply_pipeline(
            original_linear,
            analysis,
            config,
            retry_strength,
        )
        retry_validation = validate_result(
            original_linear,
            candidate_retry,
            config.budget,
        )
        if retry_validation["accepted"]:
            candidate = candidate_retry
            validation = retry_validation
        else:
            logger.warning(
                "Automatic enhancement rejected after validation: %s",
                validation["issues"],
            )
            return original.copy()

    result = rgb_float_to_bgr8(linear_to_srgb(candidate))
    return result


# ============================================================
# DIAGNOSTICS
# ============================================================


def enhancement_preview(
    image: np.ndarray,
    analysis: Mapping[str, Any],
    cfg: EnhancementConfig | None = None,
    strength: float = 1.0,
) -> Dict[str, Any]:
    """Run enhancement and return useful machine-readable diagnostics."""
    original = validate_bgr_image(image)
    config = cfg or EnhancementConfig()
    result = enhance(original, dict(analysis), config, strength)

    before_linear = srgb_to_linear(bgr8_to_rgb_float(original))
    after_linear = srgb_to_linear(bgr8_to_rgb_float(result))
    validation = validate_result(before_linear, after_linear, config.budget)

    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "image": result,
        "validation": validation,
        "strength": float(strength),
    }


# ============================================================
# JSON / FILE HELPERS
# ============================================================


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands common NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save_image(img: np.ndarray, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), validate_bgr_image(img)):
        raise EnhancementError(f"Unable to save enhanced image: {output}")


def save_validation_json(validation: Mapping[str, Any], path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(validation), cls=NumpyEncoder, indent=2),
        encoding="utf-8",
    )


# ============================================================
# BACKWARDS-COMPATIBILITY HELPERS
# ============================================================


def _apply_auto_lighting(
    rgb: np.ndarray,
    analysis: Mapping[str, Any],
    strength: float,
) -> np.ndarray:
    """Compatibility wrapper.

    Older callers may still call _apply_auto_lighting() with sRGB RGB data.
    The new implementation routes it through the linear-light tone pipeline.
    """
    linear = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    config = EnhancementConfig(
        enable_wb=False,
        enable_color_tone=False,
        enable_vibrance=False,
        enable_clarity=False,
        enable_sharpen=False,
    )
    result = _apply_pipeline(linear, analysis, config, clamp(strength, 0.0, 1.0))
    return linear_to_srgb(result)


def _apply_local_neutral_tone(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Compatibility helper for conservative residual neutralization."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    rgb8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    chroma_a = lab[:, :, 1].astype(np.float32) - 128.0
    chroma_b = lab[:, :, 2].astype(np.float32) - 128.0
    lightness = lab[:, :, 0].astype(np.float32)

    neutral = (
        (lightness >= 25)
        & (lightness <= 245)
        & (np.abs(chroma_a) <= 24)
        & (np.abs(chroma_b) <= 24)
    )

    if int(np.count_nonzero(neutral)) < 100:
        return rgb

    cast_a = float(np.median(chroma_a[neutral]))
    cast_b = float(np.median(chroma_b[neutral]))
    correction = clamp(strength * 0.50, 0.0, 1.0)

    corrected = lab.astype(np.float32)
    corrected[:, :, 1] = np.clip(
        corrected[:, :, 1] - cast_a * correction,
        0.0,
        255.0,
    )
    corrected[:, :, 2] = np.clip(
        corrected[:, :, 2] - cast_b * correction,
        0.0,
        255.0,
    )

    return np.clip(
        cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0,
        0.0,
        1.0,
    )


# ============================================================
# MODULE SELF-TEST
# ============================================================


def self_test() -> Dict[str, Any]:
    """Small deterministic test for CI/package validation."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[:, :, 2] = 150  # BGR red-biased test image.

    analysis: Dict[str, Any] = {
        "schema_version": "5.0",
        "correction_plan": {
            "white_balance": {
                "gains_rgb": [0.95, 1.0, 1.08],
                "confidence": 0.90,
            },
            "exposure_ev": 0.10,
            "highlights": -4,
            "shadows": 5,
            "contrast": 3,
            "whites": 0,
            "blacks": 0,
        },
    }

    output = enhance(image, analysis, strength=1.0)
    assert output.shape == image.shape
    assert output.dtype == np.uint8

    return {"success": True, "shape": list(output.shape), "dtype": str(output.dtype)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(self_test(), indent=2))