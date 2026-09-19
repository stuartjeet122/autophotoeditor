from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from color_constancy import (
    SelectiveMidtoneEnhancement,
)

from autophotoeditor.core.image_io import decode_color


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("image_analysis_v3")


@dataclass(frozen=True)
class EnhancementConfig:
    wb_strength: float = 1.0
    clarity_amount: float = 0.0
    enable_wb: bool = True
    enable_tone: bool = True
    enable_highlight_protection: bool = True
    enable_shadow_protection: bool = True
    enable_clarity: bool = False
    enable_vibrance: bool = False
    enable_sharpen: bool = False


class EnhancementError(Exception):
    pass


def _apply_local_neutral_tone(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Reduce a measured color cast without changing image luminosity."""
    rgb8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    chroma_a = lab[:, :, 1].astype(np.float32) - 128.0
    chroma_b = lab[:, :, 2].astype(np.float32) - 128.0
    lightness = lab[:, :, 0]
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
    correction = np.clip(strength * 0.85, 0.0, 1.0)
    corrected = lab.copy()
    corrected[:, :, 1] = np.clip(
        corrected[:, :, 1].astype(np.float32) - cast_a * correction,
        0.0,
        255.0,
    ).astype(np.uint8)
    corrected[:, :, 2] = np.clip(
        corrected[:, :, 2].astype(np.float32) - cast_b * correction,
        0.0,
        255.0,
    ).astype(np.uint8)
    return np.clip(
        cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0,
        0.0,
        1.0,
    )


def _apply_library_luminosity(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Use the third-party curve for light only, preserving source chroma."""
    library_result = SelectiveMidtoneEnhancement(
        auto=True,
        contrast_strength=0.85 * strength,
        saturation_gain=1.0,
        shadow_protection=0.12,
        highlight_protection=0.12,
    ).process(rgb)
    source_luma = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    result_luma = (
        0.2126 * library_result[:, :, 0]
        + 0.7152 * library_result[:, :, 1]
        + 0.0722 * library_result[:, :, 2]
    )
    ratio = result_luma / np.maximum(source_luma, 1e-4)
    ratio = np.clip(1.0 + (ratio - 1.0) * strength, 0.7, 1.35)
    return np.clip(rgb * ratio[:, :, None], 0.0, 1.0)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _get_edit_plan(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = _as_mapping(analysis.get("correction_plan"))
    if plan:
        return plan

    return _as_mapping(analysis.get("recommendations"))


def _get_recommendation_value(
    analysis: Mapping[str, Any],
    section: str,
    default: float = 0.0,
) -> float:
    plan = _get_edit_plan(analysis)
    value = plan.get(section, default)
    if section not in plan:
        value = _as_mapping(plan.get("light")).get(section, default)
    if isinstance(value, Mapping):
        value = value.get("value", value.get("recommended_change_ev", default))

    if value == default:
        best = analysis.get("best_enhancement") if isinstance(analysis, Mapping) else None
        if isinstance(best, Mapping):
            luminosity = _as_mapping(best.get("luminosity"))
            neutral = _as_mapping(best.get("neutral_tone"))
            if section == "exposure_ev":
                value = luminosity.get("recommended_change_ev", neutral.get("temperature_cast", default))
            elif section == "highlights":
                value = luminosity.get("highlights", default)
            elif section == "shadows":
                value = luminosity.get("shadows", default)

    return _number(value, default)


def _apply_auto_lighting(
    rgb: np.ndarray,
    analysis: Mapping[str, Any],
    strength: float,
) -> np.ndarray:
    """Apply Lightroom-style auto exposure, highlights, and shadow shaping."""
    exposure_ev = float(np.clip(_get_recommendation_value(analysis, "exposure_ev"), -1.0, 1.0))

    # Positive exposure is deliberately gentler because the library tone pass
    # below also lifts luminance and can otherwise clip bright source pixels.
    exposure_strength = strength * (0.58 if exposure_ev > 0.0 else 0.82)
    result = np.clip(rgb * (2.0 ** (exposure_ev * exposure_strength)), 0.0, 1.0)

    luminance = (
        0.2126 * result[:, :, 0]
        + 0.7152 * result[:, :, 1]
        + 0.0722 * result[:, :, 2]
    )
    highlights = float(
        np.clip(_get_recommendation_value(analysis, "highlights"), -55.0, 0.0)
    ) / 100.0
    shadows = float(
        np.clip(_get_recommendation_value(analysis, "shadows"), 0.0, 45.0)
    ) / 100.0

    highlight_weight = np.clip((luminance - 0.45) / 0.55, 0.0, 1.0)
    highlight_weight *= highlight_weight
    shadow_weight = np.clip((0.48 - luminance) / 0.48, 0.0, 1.0)
    shadow_weight = shadow_weight ** 1.45
    shadow_headroom = np.clip((0.78 - luminance) / 0.78, 0.0, 1.0)
    result += (
        (highlights * 0.22 * highlight_weight + shadows * 0.16 * shadow_weight * shadow_headroom)
        * strength
    )[:, :, None]

    contrast = float(np.clip(_get_recommendation_value(analysis, "contrast"), -20.0, 25.0)) / 100.0
    if abs(contrast) > 0.001:
        midpoint = 0.5
        result = np.clip((result - midpoint) * (1.0 + contrast * strength) + midpoint, 0.0, 1.0)

    whites = float(np.clip(_get_recommendation_value(analysis, "whites"), -25.0, 20.0)) / 100.0
    blacks = float(np.clip(_get_recommendation_value(analysis, "blacks"), -25.0, 20.0)) / 100.0
    luminance = (
        0.2126 * result[:, :, 0]
        + 0.7152 * result[:, :, 1]
        + 0.0722 * result[:, :, 2]
    )
    result += (
        whites * np.clip((luminance - 0.55) / 0.45, 0.0, 1.0)
        + blacks * np.clip((0.45 - luminance) / 0.45, 0.0, 1.0)
    )[:, :, None] * strength * 0.18
    # Keep the recovered shadows from flattening bright detail and preserve
    # highlight headroom after all tone stages have been combined.
    result = np.clip(result, 0.0, 1.0)
    luminance = (
        0.2126 * result[:, :, 0]
        + 0.7152 * result[:, :, 1]
        + 0.0722 * result[:, :, 2]
    )
    highlight_scale = np.minimum(
        1.0,
        0.985 / np.maximum(luminance, 0.985),
    )
    return np.clip(result * highlight_scale[:, :, None], 0.0, 1.0)


def enhance(
    image: np.ndarray,
    analysis: dict[str, Any],
    cfg: EnhancementConfig | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a bounded photographic enhancement to an analyzed BGR image."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise EnhancementError("Expected a 3-channel image.")
    if not isinstance(analysis, dict):
        raise EnhancementError("Analysis must be a JSON object.")
    if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
        raise EnhancementError("Strength must be between 0 and 2.")

    config = cfg or EnhancementConfig()
    original = np.clip(image, 0, 255).astype(np.uint8)
    effective_strength = min(strength, 1.0)
    if effective_strength <= 0.0:
        return original.copy()

    rgb = original[:, :, ::-1].astype(np.float32) / 255.0
    enhanced = rgb

    if config.enable_wb:
        white_balance = _as_mapping(analysis.get("white_balance"))
        plan = _get_edit_plan(analysis)
        plan_wb = _as_mapping(plan.get("white_balance"))
        correction = np.asarray(
            plan_wb.get("gains_rgb", white_balance.get("gains_rgb", white_balance.get("correction_gains_rgb", []))),
            dtype=np.float32,
        )
        planned_gains = correction.size == 3 and np.all(np.isfinite(correction)) and bool(plan_wb)
        coverage = float(
            white_balance.get("neutral_coverage", white_balance.get("neutral_pixel_coverage_percentage", 0.0))
        )
        confidence = float(
            plan_wb.get("confidence", _as_mapping(white_balance.get("estimated_temperature_cast")).get("confidence", 0.0))
        )

        best = analysis.get("best_enhancement", {}) if isinstance(analysis, Mapping) else {}
        if not (correction.size == 3 and np.all(np.isfinite(correction))):
            correction = np.asarray([], dtype=np.float32)
        if correction.size != 3 and isinstance(best, Mapping):
            neutral = best.get("neutral_tone", {})
            if isinstance(neutral, Mapping):
                temp_cast = _number(neutral.get("temperature_cast"))
                tint_cast = _number(neutral.get("tint_cast"))
                neutral_conf = _number(neutral.get("confidence"))
                if abs(temp_cast) > 0.0 or abs(tint_cast) > 0.0:
                    # The neutral-tone recommendation is used as a practical recovery signal.
                    # Act on it directly to pull the image toward a neutral balance without
                    # making the exposure stage overly aggressive. Positive temperature is
                    # treated as a cool-biased source here, so we increase red and reduce blue.
                    correction = np.array([
                        1.0 + (temp_cast * 0.04 * effective_strength) - (tint_cast * 0.02 * effective_strength),
                        1.0 + (tint_cast * 0.018 * effective_strength),
                        1.0 - (temp_cast * 0.04 * effective_strength) + (tint_cast * 0.02 * effective_strength),
                    ], dtype=np.float32)
                    coverage = max(coverage, 1.0)
                    confidence = max(confidence, neutral_conf)

        if (
            correction.size == 3
            and np.all(np.isfinite(correction))
            and (coverage >= 0.5 or planned_gains)
            and confidence >= 0.35
        ):
            correction = np.clip(correction, 0.75, 1.35)
            correction = 1.0 + (correction - 1.0) * effective_strength * confidence
            enhanced = np.clip(
                enhanced * correction.reshape(1, 1, 3),
                0.0,
                1.0,
            )

    if config.enable_tone:
        enhanced = _apply_library_luminosity(enhanced, effective_strength)

    enhanced = _apply_auto_lighting(enhanced, analysis, effective_strength)

    result = np.clip(enhanced[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    if effective_strength < 0.999:
        result = cv2.addWeighted(
            original,
            1.0 - effective_strength,
            result,
            effective_strength,
            0.0,
        )
    return result


def _save_image(img: np.ndarray, path: Path) -> None:
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise EnhancementError("Enhanced image has an invalid shape.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.clip(img, 0, 255).astype(np.uint8)):
        raise EnhancementError(f"Unable to save enhanced image: {output}")


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8
SCHEMA_VERSION = "3.0"
DEFAULT_ANALYSIS_MAX_DIMENSION = 1800
DEFAULT_DETAIL_MAX_DIMENSION = 1600


# ============================================================
# JSON
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


# ============================================================
# DATA TYPES
# ============================================================

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

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


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


def normalized_histogram(channel: np.ndarray, bins: int = 256, value_range: Tuple[float, float] = (0, 256)) -> list[float]:
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

def bgr8_to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    ).astype(np.float32)


def luminance_to_ev(y: np.ndarray, reference: float = 0.18) -> np.ndarray:
    return np.log2(np.maximum(y, 1e-6) / reference)


def chromaticity_rg(rgb_linear: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = np.sum(rgb_linear, axis=2) + EPS
    r = rgb_linear[:, :, 0] / total
    g = rgb_linear[:, :, 1] / total
    return r, g


# ============================================================
# OPTIONAL MASK INPUT
# ============================================================

def load_external_masks(mask_json_path: Optional[str], target_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Loads masks from a JSON file when provided.

    Supported forms per entry:
      "face": "C:/.../face.png"
      "person": {"path": "C:/.../person.png"}
      "sky": {"mask_path": "C:/.../sky.png"}

    Missing or invalid masks are ignored. The analyzer does not require them.
    """
    if not mask_json_path:
        return {}

    path = Path(mask_json_path)
    if not path.exists():
        logger.warning("Mask JSON does not exist: %s", path)
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unable to read mask JSON %s: %s", path, exc)
        return {}

    if isinstance(payload, Mapping) and isinstance(payload.get("masks"), Mapping):
        payload = payload["masks"]

    if not isinstance(payload, Mapping):
        return {}

    base = path.parent
    result: Dict[str, np.ndarray] = {}

    for name, value in payload.items():
        mask_path: Optional[str] = None
        if isinstance(value, str):
            mask_path = value
        elif isinstance(value, Mapping):
            for key in ("path", "mask_path", "file", "filename"):
                if isinstance(value.get(key), str):
                    mask_path = str(value[key])
                    break

        if not mask_path:
            continue

        mp = Path(mask_path)
        if not mp.is_absolute():
            mp = base / mp
        if not mp.exists():
            continue

        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        result[str(name).lower()] = ensure_mask(mask, target_shape)

    return result


