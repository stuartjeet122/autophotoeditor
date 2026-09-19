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

from autophotoeditor.core.image_io import decode_color


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("image_analysis_v3")


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


# ============================================================
# SALIENCY / SUBJECT ESTIMATION
# ============================================================

def spectral_residual_saliency(bgr: np.ndarray) -> np.ndarray:
    """OpenCV-only spectral residual saliency fallback."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    small_w = max(32, min(128, w))
    small_h = max(32, min(128, h))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    dft = cv2.dft(small, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
    log_mag = np.log(mag + EPS)
    avg = cv2.boxFilter(log_mag, -1, (3, 3), normalize=True)
    residual = log_mag - avg
    exp_mag = np.exp(residual)
    real, imag = cv2.polarToCart(exp_mag, phase)
    spec = np.dstack([real, imag]).astype(np.float32)
    inv = cv2.idft(spec, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    sal = inv * inv
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)
    sal -= float(sal.min())
    maxv = float(sal.max())
    if maxv > EPS:
        sal /= maxv
    return np.clip(sal, 0.0, 1.0).astype(np.float32)


def center_prior(shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    sx = max(w * 0.32, 1.0)
    sy = max(h * 0.32, 1.0)
    g = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    return g.astype(np.float32)


def derive_subject_mask(
    bgr: np.ndarray,
    external_masks: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    h, w = bgr.shape[:2]

    for key in ("subject", "person", "foreground", "main_subject"):
        if key in external_masks:
            m = ensure_mask(external_masks[key], (h, w))
            cov = mask_coverage(m)
            conf = clamp(0.78 + min(cov, 40.0) / 200.0, 0.0, 0.98)
            return m, m.astype(np.float32), f"external:{key}", conf

    sal = spectral_residual_saliency(bgr)
    prior = center_prior((h, w))
    combined = 0.72 * sal + 0.28 * prior

    threshold = max(float(np.percentile(combined, 72)), 0.38)
    mask = combined >= threshold

    # Clean small isolated patches.
    u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = u8 > 0

    cov = mask_coverage(mask)
    sal_strength = float(np.mean(combined[mask])) if np.any(mask) else 0.0
    conf = clamp(0.35 + 0.45 * sal_strength + 0.2 * (1.0 if 3.0 <= cov <= 65.0 else 0.3), 0.1, 0.82)
    return mask, combined.astype(np.float32), "saliency", conf


# ============================================================
# SEMANTIC COLOR HEURISTICS
# ============================================================

def estimate_semantic_color_masks(bgr: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Lightweight semantic-ish masks used only as supporting evidence.
    They are deliberately conservative and are NOT treated as ground truth.
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    L = lab[:, :, 0]
    A = lab[:, :, 1].astype(np.int16) - 128
    B = lab[:, :, 2].astype(np.int16) - 128

    # Vegetation: mostly green/cyan-green and reasonably saturated.
    vegetation = (H >= 30) & (H <= 95) & (S >= 45) & (V >= 30)

    # Sky: blue/cyan, typically top-biased; this is supporting evidence only.
    h, w = H.shape
    yy = np.arange(h)[:, None]
    top_bias = yy < int(h * 0.75)
    sky = (H >= 85) & (H <= 135) & (S >= 30) & (V >= 70) & top_bias

    # Skin heuristic in YCrCb is usually more robust than raw HSV alone.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin = (
        (Y >= 35)
        & (Cr >= 133) & (Cr <= 180)
        & (Cb >= 75) & (Cb <= 135)
        & (S >= 15) & (S <= 190)
    )

    # Neutral candidates - broad; stricter scoring happens in WB analysis.
    neutral = (S <= 50) & (V >= 40) & (V <= 245) & (L >= 35) & (L <= 245)

    # Warm-source-colored pixels, useful to avoid over-correcting intentional tungsten/neon.
    warm = (B >= 10) & (A >= -5) & (V >= 45)

    return {
        "vegetation": vegetation,
        "sky": sky,
        "skin": skin,
        "neutral_candidates": neutral,
        "warm_pixels": warm,
    }


# ============================================================
# TONE / EXPOSURE
# ============================================================

def analyze_tone(
    bgr: np.ndarray,
    lab: np.ndarray,
    linear_y: np.ndarray,
    subject_mask: np.ndarray,
    saliency_map: np.ndarray,
) -> Dict[str, Any]:
    L = lab[:, :, 0].astype(np.float32)

    percentiles = percentile_dict(
        L,
        (0.1, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.9),
    )
    p01 = percentiles["p1"] if "p1" in percentiles else percentiles.get("p1_0", 0.0)
    p99 = percentiles["p99"] if "p99" in percentiles else percentiles.get("p99_0", 255.0)

    any_high_clip = np.any(bgr >= 254, axis=2)
    all_high_clip = np.all(bgr >= 254, axis=2)
    any_low_clip = np.any(bgr <= 1, axis=2)
    all_low_clip = np.all(bgr <= 1, axis=2)

    # Linear-light exposure evidence.
    y = np.clip(linear_y.astype(np.float32), 0.0, 1.0)
    ev = luminance_to_ev(y)

    valid = (y > 0.001) & (y < 0.98)
    global_ev = robust_median(ev[valid], -3.0)

    subject_valid = subject_mask & valid
    subject_ev = robust_median(ev[subject_valid], global_ev)

    # Saliency-weighted luminance provides an intent-aware middle ground.
    sw = np.clip(saliency_map.astype(np.float64), 0.0, 1.0)
    valid_w = sw * valid.astype(np.float64)
    if float(valid_w.sum()) > EPS:
        weighted_y = float(np.sum(y * valid_w) / np.sum(valid_w))
        salient_ev = math.log2(max(weighted_y, 1e-6) / 0.18)
    else:
        salient_ev = global_ev

    # Desired subject midpoint is intentionally conservative. We are not assuming
    # every scene should average to 18% gray.
    subject_cov = mask_coverage(subject_mask) / 100.0
    global_mid = float(np.median(y))
    subject_mid = float(np.median(y[subject_mask])) if np.any(subject_mask) else global_mid

    # Highlight headroom in linear light.
    y99 = float(np.percentile(y, 99.0))
    y999 = float(np.percentile(y, 99.9))
    highlight_headroom_ev = math.log2(1.0 / max(y99, 1e-6))

    mean_l = float(np.mean(L))
    median_l = float(np.median(L))
    std_l = float(np.std(L))
    dark_pct = float(np.mean(L <= 25) * 100.0)
    very_dark_pct = float(np.mean(L <= 10) * 100.0)
    bright_pct = float(np.mean(L >= 230) * 100.0)
    very_bright_pct = float(np.mean(L >= 245) * 100.0)
    high_any_pct = float(np.mean(any_high_clip) * 100.0)
    high_all_pct = float(np.mean(all_high_clip) * 100.0)
    low_any_pct = float(np.mean(any_low_clip) * 100.0)
    low_all_pct = float(np.mean(all_low_clip) * 100.0)

    return {
        # v2 compatibility fields
        "mean_luminance": round(mean_l, 4),
        "median_luminance": round(median_l, 4),
        "contrast_rms": round(std_l, 4),
        "percentiles": percentiles,
        "effective_dynamic_range": round(float(p99 - p01), 4),
        "dark_pixels_percentage": round(dark_pct, 4),
        "very_dark_pixels_percentage": round(very_dark_pct, 4),
        "bright_pixels_percentage": round(bright_pct, 4),
        "very_bright_pixels_percentage": round(very_bright_pct, 4),
        "highlight_clipping": {
            "any_channel_percentage": round(high_any_pct, 5),
            "all_channels_percentage": round(high_all_pct, 5),
        },
        "shadow_clipping": {
            "any_channel_percentage": round(low_any_pct, 5),
            "all_channels_percentage": round(low_all_pct, 5),
        },
        "is_clipped_highlights": high_all_pct > 0.5,
        "is_clipped_shadows": low_all_pct > 0.5,

        "perceptual": {
            "mean_luminance_lab": round(float(np.mean(L)), 4),
            "median_luminance_lab": round(float(np.median(L)), 4),
            "contrast_rms_lab": round(float(np.std(L)), 4),
            "percentiles_lab": percentiles,
            "effective_dynamic_range_lab": round(float(p99 - p01), 4),
            "dark_pixels_percentage": round(float(np.mean(L <= 25) * 100.0), 4),
            "very_dark_pixels_percentage": round(float(np.mean(L <= 10) * 100.0), 4),
            "bright_pixels_percentage": round(float(np.mean(L >= 230) * 100.0), 4),
            "very_bright_pixels_percentage": round(float(np.mean(L >= 245) * 100.0), 4),
        },
        "linear_light": {
            "mean_luminance": round(float(np.mean(y)), 6),
            "median_luminance": round(float(np.median(y)), 6),
            "global_median_ev_from_18pct": round(global_ev, 4),
            "subject_median_ev_from_18pct": round(subject_ev, 4),
            "saliency_weighted_ev_from_18pct": round(salient_ev, 4),
            "global_median": round(global_mid, 6),
            "subject_median": round(subject_mid, 6),
            "subject_coverage_percentage": round(subject_cov * 100.0, 4),
            "p99_luminance": round(y99, 6),
            "p99_9_luminance": round(y999, 6),
            "highlight_headroom_ev_at_p99": round(highlight_headroom_ev, 4),
        },
        "clipping": {
            "highlight_any_channel_percentage": round(float(np.mean(any_high_clip) * 100.0), 5),
            "highlight_all_channels_percentage": round(float(np.mean(all_high_clip) * 100.0), 5),
            "shadow_any_channel_percentage": round(float(np.mean(any_low_clip) * 100.0), 5),
            "shadow_all_channels_percentage": round(float(np.mean(all_low_clip) * 100.0), 5),
        },
    }


# ============================================================
# WHITE BALANCE
# ============================================================

def _channel_gains_from_estimated_illuminant(rgb: np.ndarray) -> Tuple[float, float, float]:
    r, g, b = [max(float(x), 1e-6) for x in rgb]
    target = (r + g + b) / 3.0
    return target / r, target / g, target / b


def _wb_estimator_gray_world(rgb_linear: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, float]:
    pixels = rgb_linear[valid]
    if pixels.shape[0] < 100:
        return np.array([1.0, 1.0, 1.0], np.float32), 0.0
    illum = np.mean(pixels, axis=0)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)
    spread = float(np.std(illum) / max(np.mean(illum), EPS))
    conf = clamp(0.35 + 0.35 * min(pixels.shape[0] / 10000.0, 1.0) - 0.15 * spread, 0.15, 0.7)
    return gains, conf


def _wb_estimator_shades_of_gray(rgb_linear: np.ndarray, valid: np.ndarray, p: float = 6.0) -> Tuple[np.ndarray, float]:
    pixels = rgb_linear[valid]
    if pixels.shape[0] < 100:
        return np.array([1.0, 1.0, 1.0], np.float32), 0.0
    illum = np.power(np.mean(np.power(np.maximum(pixels, 1e-6), p), axis=0), 1.0 / p)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)
    conf = clamp(0.32 + 0.35 * min(pixels.shape[0] / 10000.0, 1.0), 0.15, 0.67)
    return gains, conf


def _wb_estimator_neutral_lab(
    rgb_linear: np.ndarray,
    lab: np.ndarray,
    neutral_mask: np.ndarray,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    A = lab[:, :, 1].astype(np.float32) - 128.0
    B = lab[:, :, 2].astype(np.float32) - 128.0
    count = int(np.count_nonzero(neutral_mask))
    total = neutral_mask.size
    coverage = count / max(total, 1)

    if count < max(100, int(total * 0.0015)):
        return (
            np.array([1.0, 1.0, 1.0], np.float32),
            0.0,
            {"neutral_a": 0.0, "neutral_b": 0.0, "coverage_percentage": coverage * 100.0},
        )

    pixels = rgb_linear[neutral_mask]
    illum = np.median(pixels, axis=0)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)

    neutral_a = robust_median(A[neutral_mask])
    neutral_b = robust_median(B[neutral_mask])
    neutrality = math.exp(-(abs(neutral_a) + abs(neutral_b)) / 22.0)
    coverage_score = min(coverage / 0.12, 1.0)
    conf = clamp(0.25 + 0.5 * coverage_score + 0.2 * neutrality, 0.2, 0.93)

    return gains, conf, {
        "neutral_a": neutral_a,
        "neutral_b": neutral_b,
        "coverage_percentage": coverage * 100.0,
    }


def gains_to_cast(gains: np.ndarray) -> Tuple[float, float]:
    """
    Maps RGB correction gains to stable relative warm/cool and green/magenta
    indicators. These are edit-direction indicators, not camera Kelvin values.
    """
    r, g, b = [max(float(x), 1e-6) for x in gains]
    # If correction needs more blue than red, source is warm => positive cast.
    temperature_cast = math.log2(b / r) * 12.0
    # If correction needs more green relative to R/B average, source is magenta.
    tint_cast = math.log2(g / math.sqrt(r * b)) * 12.0
    return temperature_cast, tint_cast


def analyze_white_balance(
    bgr: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    rgb_linear: np.ndarray,
    semantic_masks: Mapping[str, np.ndarray],
    external_masks: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    H, S, V = cv2.split(hsv)
    L = lab[:, :, 0]

    valid = (V >= 35) & (V <= 245) & (L >= 30) & (L <= 245)

    # Exclude colors that are poor neutral references.
    exclusion = np.zeros(valid.shape, dtype=bool)
    for key in ("skin", "vegetation", "sky"):
        if key in semantic_masks:
            exclusion |= semantic_masks[key]
        if key in external_masks:
            exclusion |= ensure_mask(external_masks[key], valid.shape)

    neutral_mask = valid & (S <= 48) & (~exclusion)

    gw_gains, gw_conf = _wb_estimator_gray_world(rgb_linear, valid & (~exclusion))
    sg_gains, sg_conf = _wb_estimator_shades_of_gray(rgb_linear, valid & (~exclusion), p=6.0)
    nl_gains, nl_conf, nl_meta = _wb_estimator_neutral_lab(rgb_linear, lab, neutral_mask)

    estimators = [
        ("neutral_lab", nl_gains, nl_conf),
        ("gray_world", gw_gains, gw_conf),
        ("shades_of_gray", sg_gains, sg_conf),
    ]

    valid_est = [(n, g, c) for n, g, c in estimators if c > 0.0]
    if valid_est:
        weights = np.array([c for _, _, c in valid_est], dtype=np.float64)
        log_gains = np.vstack([np.log(np.maximum(g, 1e-6)) for _, g, _ in valid_est])
        fused = np.exp(np.average(log_gains, axis=0, weights=weights)).astype(np.float32)

        # Agreement: how much estimators disagree in log-gain space.
        dispersion = float(np.mean(np.std(log_gains, axis=0))) if len(valid_est) > 1 else 0.0
        agreement = math.exp(-dispersion * 4.0)
        base_conf = float(np.average(weights, weights=weights))
        fused_conf = clamp(base_conf * (0.65 + 0.35 * agreement), 0.0, 0.96)
    else:
        fused = np.ones(3, dtype=np.float32)
        fused_conf = 0.0
        agreement = 0.0

    # Normalize gains around green for readability.
    fused = fused / max(float(fused[1]), 1e-6)
    temp_cast, tint_cast = gains_to_cast(fused)

    return {
        "method": "multi-estimator-fusion",
        "correction_gains_rgb": [round(float(x), 6) for x in fused],
        "estimated_temperature_cast": Estimate(temp_cast, fused_conf).to_json(),
        "estimated_tint_cast": Estimate(tint_cast, fused_conf).to_json(),
        "neutral_pixel_coverage_percentage": round(float(nl_meta["coverage_percentage"]), 4),
        "neutral_lab_a_median": round(float(nl_meta["neutral_a"]), 4),
        "neutral_lab_b_median": round(float(nl_meta["neutral_b"]), 4),
        "estimator_agreement": round(float(agreement), 4),
        "estimators": {
            name: {
                "gains_rgb": [round(float(x), 6) for x in gains / max(float(gains[1]), 1e-6)],
                "confidence": round(float(conf), 4),
            }
            for name, gains, conf in estimators
        },
        "interpretation": {
            "temperature": "warm" if temp_cast > 2.0 else "cool" if temp_cast < -2.0 else "neutral",
            "tint": "magenta" if tint_cast > 2.0 else "green" if tint_cast < -2.0 else "neutral",
        },
        "warning": "Temperature/tint are image-cast indicators, not camera Kelvin/tint metadata.",
    }


def analyze_color_profile(
    bgr: np.ndarray,
    hsv: np.ndarray,
    wb: Mapping[str, Any],
) -> Dict[str, Any]:
    B = bgr[:, :, 0].astype(np.float32)
    G = bgr[:, :, 1].astype(np.float32)
    R = bgr[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1]

    sat_percentiles = percentile_dict(S, (5, 10, 25, 50, 75, 90, 95, 99))
    mean_b, mean_g, mean_r = float(np.mean(B)), float(np.mean(G)), float(np.mean(R))

    # Compatibility representation: scalar cast values are kept here while the
    # richer confidence-aware estimates remain in top-level white_balance.
    compat_wb = {
        "neutral_pixel_coverage_percentage": wb["neutral_pixel_coverage_percentage"],
        "neutral_lab_a_median": wb["neutral_lab_a_median"],
        "neutral_lab_b_median": wb["neutral_lab_b_median"],
        "estimated_tint_cast": wb["estimated_tint_cast"]["value"],
        "estimated_temperature_cast": wb["estimated_temperature_cast"]["value"],
        "confidence": wb["estimated_temperature_cast"]["confidence"],
        "interpretation": wb["interpretation"],
    }

    return {
        "mean_bgr": [round(mean_b, 4), round(mean_g, 4), round(mean_r, 4)],
        "std_bgr": [
            round(float(np.std(B)), 4),
            round(float(np.std(G)), 4),
            round(float(np.std(R)), 4),
        ],
        "saturation_mean": round(float(np.mean(S)), 4),
        "saturation_std": round(float(np.std(S)), 4),
        "saturation_percentiles": sat_percentiles,
        "low_saturation_percentage": round(float(np.mean(S < 25) * 100.0), 4),
        "medium_saturation_percentage": round(float(np.mean((S >= 25) & (S < 100)) * 100.0), 4),
        "high_saturation_percentage": round(float(np.mean(S >= 180) * 100.0), 4),
        "extreme_saturation_percentage": round(float(np.mean(S >= 220) * 100.0), 4),
        "white_balance": compat_wb,
        "diagnostic_rgb_ratios": {
            "red_blue_ratio": round(mean_r / max(mean_b, EPS), 6),
            "red_green_ratio": round(mean_r / max(mean_g, EPS), 6),
            "blue_green_ratio": round(mean_b / max(mean_g, EPS), 6),
        },
    }


# ============================================================
# CONTRAST / DETAIL / NOISE
# ============================================================

def analyze_contrast(lab_l: np.ndarray, linear_y: np.ndarray) -> Dict[str, float]:
    L = lab_l.astype(np.float32)
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=15, sigmaY=15)
    local_difference = L - blurred

    y = np.clip(linear_y.astype(np.float32), 0.0, 1.0)
    log_y = np.log2(np.maximum(y, 1e-5))

    return {
        "global_std_lab": round(float(np.std(L)), 4),
        "local_std_lab": round(float(np.std(local_difference)), 4),
        "rms_lab": round(float(np.sqrt(np.mean(np.square(L - np.mean(L))))), 4),
        "log_luminance_std_ev": round(float(np.std(log_y)), 4),
    }


def analyze_detail(gray: np.ndarray, subject_mask: np.ndarray) -> Dict[str, Any]:
    analysis_gray = resize_longest(gray, DEFAULT_DETAIL_MAX_DIMENSION)
    h0, w0 = gray.shape
    h, w = analysis_gray.shape

    if (h, w) != (h0, w0):
        subject = cv2.resize(subject_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        subject = subject_mask

    gray_f = analysis_gray.astype(np.float32)

    lap = cv2.Laplacian(analysis_gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    tenengrad = float(np.mean(grad * grad))
    grad_median = float(np.median(grad))

    # Flat-region noise estimator using robust MAD of a high-pass residual.
    smooth_large = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
    highpass = gray_f - smooth_large

    local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    local_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    local_var = np.maximum(local_sq - local_mean * local_mean, 0.0)

    flat_threshold = float(np.percentile(local_var, 35.0))
    flat = local_var <= flat_threshold
    flat &= (analysis_gray >= 20) & (analysis_gray <= 235)

    hp_flat = highpass[flat]
    sigma = 1.4826 * robust_mad(hp_flat) if hp_flat.size >= 100 else float(np.std(highpass))

    # Motion/defocus clues from directional gradient anisotropy and edge width proxy.
    abs_gx = float(np.mean(np.abs(gx)))
    abs_gy = float(np.mean(np.abs(gy)))
    anisotropy = abs(abs_gx - abs_gy) / max(abs_gx + abs_gy, EPS)

    edges = grad > np.percentile(grad, 82)
    edge_density = float(np.mean(edges))

    if np.any(subject):
        subject_sharpness = float(np.mean(grad[subject]))
    else:
        subject_sharpness = float(np.mean(grad))

    # Confidence favors enough flat pixels for noise and enough edges for sharpness.
    flat_cov = float(np.mean(flat))
    noise_conf = clamp(0.25 + min(flat_cov / 0.25, 1.0) * 0.7, 0.2, 0.95)
    sharp_conf = clamp(0.3 + min(edge_density / 0.18, 1.0) * 0.6, 0.25, 0.93)

    return {
        "analysis_width": int(w),
        "analysis_height": int(h),
        "laplacian_variance": round(lap_var, 4),
        "tenengrad": round(tenengrad, 4),
        "gradient_median": round(grad_median, 4),
        "subject_gradient_mean": round(subject_sharpness, 4),
        "edge_density": round(edge_density, 6),
        "gradient_anisotropy": round(float(anisotropy), 6),
        "noise_sigma_flat_regions": Estimate(float(sigma), noise_conf).to_json(),
        "flat_region_coverage_percentage": round(flat_cov * 100.0, 4),
        "sharpness_confidence": round(sharp_conf, 4),
    }


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return safe_float(-np.sum(p * np.log2(p)))


# ============================================================
# REGION / SEMANTIC ANALYSIS
# ============================================================

def analyze_mask_region(
    name: str,
    mask: np.ndarray,
    lab: np.ndarray,
    hsv: np.ndarray,
    linear_y: np.ndarray,
) -> Dict[str, Any]:
    m = mask.astype(bool)
    cov = mask_coverage(m)
    if not np.any(m):
        return {"name": name, "coverage_percentage": 0.0, "available": False}

    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    y = linear_y
    vals = y[m]

    return {
        "name": name,
        "available": True,
        "coverage_percentage": round(cov, 4),
        "mean_luminance_lab": round(float(np.mean(L[m])), 4),
        "median_luminance_lab": round(float(np.median(L[m])), 4),
        "mean_saturation": round(float(np.mean(S[m])), 4),
        "median_linear_luminance": round(float(np.median(vals)), 6),
        "median_ev_from_18pct": round(float(np.median(luminance_to_ev(vals))), 4),
        "p05_linear_luminance": round(float(np.percentile(vals, 5)), 6),
        "p95_linear_luminance": round(float(np.percentile(vals, 95)), 6),
    }


def analyze_grid_regions(image: np.ndarray, rows: int = 3, cols: int = 3) -> Dict[str, Any]:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    regions: Dict[str, Any] = {}

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * h // rows, (row + 1) * h // rows
            x1, x2 = col * w // cols, (col + 1) * w // cols
            rl = L[y1:y2, x1:x2]
            rs = S[y1:y2, x1:x2]
            if rl.size == 0:
                continue
            key = f"r{row + 1}_c{col + 1}"
            regions[key] = {
                "row": row + 1,
                "column": col + 1,
                "mean_luminance": round(float(np.mean(rl)), 4),
                "median_luminance": round(float(np.median(rl)), 4),
                "contrast": round(float(np.std(rl)), 4),
                "mean_saturation": round(float(np.mean(rs)), 4),
                "dark_percentage": round(float(np.mean(rl < 25) * 100), 4),
                "bright_percentage": round(float(np.mean(rl > 230) * 100), 4),
            }
    return regions


# ============================================================
# SCENE UNDERSTANDING
# ============================================================

def classify_scene(
    semantic_masks: Mapping[str, np.ndarray],
    external_masks: Mapping[str, np.ndarray],
    subject_mask: np.ndarray,
    tone: Mapping[str, Any],
) -> Dict[str, Any]:
    def cov(name: str) -> float:
        if name in external_masks:
            return mask_coverage(external_masks[name])
        if name in semantic_masks:
            return mask_coverage(semantic_masks[name])
        return 0.0

    skin_cov = cov("skin")
    sky_cov = cov("sky")
    veg_cov = cov("vegetation")
    person_cov = cov("person")
    face_cov = cov("face")
    subject_cov = mask_coverage(subject_mask)

    med = float(tone["linear_light"]["global_median"])
    p99 = float(tone["linear_light"]["p99_luminance"])

    scores = {
        "portrait": clamp((max(person_cov, face_cov * 2.0, skin_cov) / 28.0), 0.0, 1.0),
        "landscape": clamp((sky_cov + veg_cov) / 45.0, 0.0, 1.0),
        "low_key": clamp((0.12 - med) / 0.12, 0.0, 1.0),
        "high_key": clamp((med - 0.32) / 0.35, 0.0, 1.0),
    }

    if max(scores.values()) < 0.28:
        scene_type = "general"
        confidence = 0.45
    else:
        scene_type = max(scores, key=scores.get)
        confidence = clamp(0.45 + 0.45 * scores[scene_type], 0.45, 0.9)

    return {
        "type": scene_type,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "evidence": {
            "subject_coverage_percentage": round(subject_cov, 4),
            "person_coverage_percentage": round(person_cov, 4),
            "face_coverage_percentage": round(face_cov, 4),
            "skin_coverage_percentage": round(skin_cov, 4),
            "sky_coverage_percentage": round(sky_cov, 4),
            "vegetation_coverage_percentage": round(veg_cov, 4),
            "global_median_linear_luminance": round(med, 6),
            "p99_linear_luminance": round(p99, 6),
        },
    }


# ============================================================
# OPTIONAL AI SCENE CLASSIFIER
# ============================================================

class OptionalAIAnalyzer:
    """
    Optional zero-shot CLIP scene classifier.

    It only activates when --ai is supplied and the required packages/model are
    available. The rest of the analyzer remains fully functional without it.

    Default model: openai/clip-vit-base-patch32
    """

    LABELS = [
        "portrait photo",
        "group photo",
        "landscape photo",
        "city photo",
        "indoor photo",
        "night photo",
        "food photo",
        "product photo",
        "pet photo",
        "sports photo",
        "architecture photo",
        "document photo",
        "macro photo",
    ]

    def __init__(self, enabled: bool, model_name: str):
        self.enabled = enabled
        self.model_name = model_name
        self._pipe = None
        self.error: Optional[str] = None

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self._pipe is not None:
            return True
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline(
                task="zero-shot-image-classification",
                model=self.model_name,
            )
            return True
        except Exception as exc:
            self.error = f"AI scene classifier unavailable: {exc}"
            logger.warning(self.error)
            return False

    def analyze(self, bgr: np.ndarray) -> Dict[str, Any]:
        if not self._load():
            return {
                "enabled": bool(self.enabled),
                "available": False,
                "model": self.model_name,
                "error": self.error,
            }
        try:
            from PIL import Image  # type: ignore
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            outputs = self._pipe(image, candidate_labels=self.LABELS)
            top = outputs[:5]
            return {
                "enabled": True,
                "available": True,
                "model": self.model_name,
                "top_labels": [
                    {"label": str(x["label"]), "score": round(float(x["score"]), 6)}
                    for x in top
                ],
            }
        except Exception as exc:
            return {
                "enabled": True,
                "available": False,
                "model": self.model_name,
                "error": str(exc),
            }


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def derive_exposure_recommendation(
    tone: Mapping[str, Any],
    scene: Mapping[str, Any],
    subject_confidence: float,
) -> Estimate:
    linear = tone["linear_light"]
    clip = tone["clipping"]

    global_ev = float(linear["global_median_ev_from_18pct"])
    subject_ev = float(linear["subject_median_ev_from_18pct"])
    salient_ev = float(linear["saliency_weighted_ev_from_18pct"])
    subject_cov = float(linear["subject_coverage_percentage"]) / 100.0

    # Instead of forcing every image to 18% gray, target depends on scene.
    scene_type = str(scene.get("type", "general"))
    target_global_ev = {
        "low_key": -1.35,
        "high_key": 0.75,
        "portrait": -0.25,
        "landscape": -0.45,
        "general": -0.45,
    }.get(scene_type, -0.45)

    target_subject_ev = -0.05 if scene_type == "portrait" else -0.35

    global_delta = target_global_ev - global_ev
    subject_delta = target_subject_ev - subject_ev
    salient_delta = -0.25 - salient_ev

    if subject_cov >= 0.02:
        proposed = weighted_mean(
            [global_delta, subject_delta, salient_delta],
            [0.30, 0.50 * subject_confidence, 0.20],
        )
    else:
        proposed = weighted_mean([global_delta, salient_delta], [0.65, 0.35])

    # Highlight protection. JPEG clipping cannot be recovered by exposure increase.
    high_clip = float(clip["highlight_all_channels_percentage"])
    headroom = float(linear["highlight_headroom_ev_at_p99"])
    if proposed > 0:
        safe_raise = max(0.0, headroom - 0.18)
        if high_clip > 0.4:
            safe_raise = min(safe_raise, 0.08)
        proposed = min(proposed, safe_raise)

    proposed = clamp(proposed, -1.5, 1.5)

    clipping_penalty = min(high_clip / 3.0, 1.0)
    confidence = clamp(
        0.52
        + 0.25 * subject_confidence
        + 0.13 * float(scene.get("confidence", 0.5))
        - 0.18 * clipping_penalty,
        0.25,
        0.94,
    )
    return Estimate(proposed, confidence)


def derive_edit_recommendations(
    tone: Mapping[str, Any],
    wb: Mapping[str, Any],
    contrast: Mapping[str, Any],
    detail: Mapping[str, Any],
    scene: Mapping[str, Any],
    subject_confidence: float,
) -> Dict[str, Any]:
    exposure = derive_exposure_recommendation(tone, scene, subject_confidence)

    clip = tone["clipping"]
    perceptual = tone["perceptual"]
    linear = tone["linear_light"]

    high_clip = float(clip["highlight_any_channel_percentage"])
    low_clip = float(clip["shadow_any_channel_percentage"])
    bright_pct = float(perceptual["bright_pixels_percentage"])
    dark_pct = float(perceptual["dark_pixels_percentage"])

    # Lightroom-like -100..100 advisory values, intentionally conservative.
    highlights = clamp(-(high_clip * 4.5 + max(bright_pct - 6.0, 0.0) * 1.6), -65.0, 0.0)
    shadows = clamp(low_clip * 3.5 + max(dark_pct - 10.0, 0.0) * 1.3, 0.0, 55.0)

    contrast_std = float(contrast["global_std_lab"])
    if contrast_std < 38:
        contrast_rec = clamp((38 - contrast_std) * 0.65, 0.0, 18.0)
    elif contrast_std > 62:
        contrast_rec = -clamp((contrast_std - 62) * 0.35, 0.0, 12.0)
    else:
        contrast_rec = 0.0

    noise_sigma = float(detail["noise_sigma_flat_regions"]["value"])
    noise_conf = float(detail["noise_sigma_flat_regions"]["confidence"])
    noise_reduction = clamp((noise_sigma - 1.8) * 3.8, 0.0, 35.0) * noise_conf

    subject_grad = float(detail["subject_gradient_mean"])
    sharp_conf = float(detail["sharpness_confidence"])
    sharpen = clamp((18.0 - subject_grad) * 1.5, 0.0, 22.0) * sharp_conf

    temp = float(wb["estimated_temperature_cast"]["value"])
    tint = float(wb["estimated_tint_cast"]["value"])
    wb_conf = float(wb["estimated_temperature_cast"]["confidence"])

    # Slider directions oppose the detected cast.
    temperature_adjust = clamp(-temp * 1.6 * wb_conf, -22.0, 22.0)
    tint_adjust = clamp(-tint * 1.6 * wb_conf, -22.0, 22.0)

    recommendation_confidence = clamp(
        weighted_mean(
            [exposure.confidence, wb_conf, noise_conf, sharp_conf, float(scene.get("confidence", 0.5))],
            [0.35, 0.2, 0.1, 0.1, 0.25],
        ),
        0.0,
        1.0,
    )

    return {
        "confidence": round(recommendation_confidence, 4),
        "exposure_ev": exposure.to_json(),
        "highlights": round(highlights, 3),
        "shadows": round(shadows, 3),
        "contrast": round(contrast_rec, 3),
        "temperature": {
            "value": round(temperature_adjust, 3),
            "confidence": round(wb_conf, 4),
        },
        "tint": {
            "value": round(tint_adjust, 3),
            "confidence": round(wb_conf, 4),
        },
        "noise_reduction": {
            "value": round(noise_reduction, 3),
            "confidence": round(noise_conf, 4),
        },
        "sharpen": {
            "value": round(sharpen, 3),
            "confidence": round(sharp_conf, 4),
        },
        "notes": [
            "Recommendations are conservative edit suggestions, not absolute photographic truth.",
            "Temperature/tint are derived from image cast, not camera RAW Kelvin metadata.",
            "For best results, semantic person/face/sky masks should be supplied by the existing mask pipeline.",
        ],
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_image(
    tone: Mapping[str, Any],
    wb: Mapping[str, Any],
    detail: Mapping[str, Any],
    scene: Mapping[str, Any],
    recommendations: Mapping[str, Any],
) -> Dict[str, Any]:
    exp = float(recommendations["exposure_ev"]["value"])
    exp_conf = float(recommendations["exposure_ev"]["confidence"])

    if exp > 0.55:
        exposure_status = "underexposed"
    elif exp > 0.18:
        exposure_status = "slightly_underexposed"
    elif exp < -0.55:
        exposure_status = "overexposed"
    elif exp < -0.18:
        exposure_status = "slightly_overexposed"
    else:
        exposure_status = "balanced_for_detected_scene"

    clip = tone["clipping"]
    hclip = float(clip["highlight_all_channels_percentage"])
    sclip = float(clip["shadow_all_channels_percentage"])

    highlight_status = "severe_clipping" if hclip > 2.0 else "clipping" if hclip > 0.5 else "normal"
    shadow_status = "severe_clipping" if sclip > 2.0 else "clipping" if sclip > 0.5 else "normal"

    noise = float(detail["noise_sigma_flat_regions"]["value"])
    noise_status = "very_low" if noise < 1.5 else "low" if noise < 3.0 else "moderate" if noise < 6.0 else "high"

    grad = float(detail["subject_gradient_mean"])
    sharpness_status = "very_soft" if grad < 5 else "soft" if grad < 10 else "normal" if grad < 24 else "high_detail"

    return {
        "scene": scene,
        "exposure_status": exposure_status,
        "exposure": {
            "status": exposure_status,
            "confidence": round(exp_conf, 4),
        },
        "highlight_status": highlight_status,
        "shadow_status": shadow_status,
        "sharpness_status": sharpness_status,
        "noise_status": noise_status,
        "color_cast": wb["interpretation"],
    }


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_image_data(
    image_path: str,
    analysis_max_dimension: int = DEFAULT_ANALYSIS_MAX_DIMENSION,
    mask_json_path: Optional[str] = None,
    enable_ai: bool = False,
    ai_model: str = "openai/clip-vit-base-patch32",
) -> Dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    try:
        img = decode_color(os.path.abspath(str(path)))
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc

    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Image decoder returned an invalid image.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {img.shape}.")
    if img.dtype != np.uint8:
        # Keep behavior predictable for this analyzer. RAW/16-bit should be handled
        # by a dedicated linear pipeline rather than silently cast here.
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w, c = img.shape
    file_size = path.stat().st_size
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    analysis_img = resize_longest(img, analysis_max_dimension)
    ah, aw = analysis_img.shape[:2]

    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)

    rgb = bgr8_to_rgb_float(analysis_img)
    rgb_linear = srgb_to_linear(rgb)
    linear_y = linear_luminance(rgb_linear)

    external_masks = load_external_masks(mask_json_path, (ah, aw))
    semantic_masks = estimate_semantic_color_masks(analysis_img, hsv, lab)
    subject_mask, saliency_map, subject_method, subject_confidence = derive_subject_mask(
        analysis_img,
        external_masks,
    )

    tone = analyze_tone(analysis_img, lab, linear_y, subject_mask, saliency_map)
    wb = analyze_white_balance(
        analysis_img,
        hsv,
        lab,
        rgb_linear,
        semantic_masks,
        external_masks,
    )
    color_profile = analyze_color_profile(analysis_img, hsv, wb)
    contrast = analyze_contrast(lab[:, :, 0], linear_y)
    detail = analyze_detail(gray, subject_mask)
    entropy = calculate_entropy(gray)

    scene = classify_scene(semantic_masks, external_masks, subject_mask, tone)

    semantic_regions: Dict[str, Any] = {
        "subject": analyze_mask_region("subject", subject_mask, lab, hsv, linear_y)
    }

    # Prefer externally supplied masks when available, otherwise use conservative heuristics.
    region_names = sorted(set(semantic_masks.keys()) | set(external_masks.keys()))
    for name in region_names:
        if name in ("neutral_candidates", "warm_pixels"):
            continue
        mask = external_masks.get(name, semantic_masks.get(name))
        if mask is not None:
            semantic_regions[name] = analyze_mask_region(name, ensure_mask(mask, (ah, aw)), lab, hsv, linear_y)

    grid_regions = analyze_grid_regions(analysis_img, rows=3, cols=3)

    recommendations = derive_edit_recommendations(
        tone,
        wb,
        contrast,
        detail,
        scene,
        subject_confidence,
    )

    classification = classify_image(tone, wb, detail, scene, recommendations)

    ai = OptionalAIAnalyzer(enable_ai, ai_model)
    ai_result = ai.analyze(analysis_img)

    histograms = {
        "blue_256bins_normalized": normalized_histogram(analysis_img[:, :, 0]),
        "green_256bins_normalized": normalized_histogram(analysis_img[:, :, 1]),
        "red_256bins_normalized": normalized_histogram(analysis_img[:, :, 2]),
        "luminance_lab_256bins_normalized": normalized_histogram(lab[:, :, 0]),
        "saturation_256bins_normalized": normalized_histogram(hsv[:, :, 1]),
        "linear_luminance_128bins_normalized": normalized_histogram(linear_y, bins=128, value_range=(0.0, 1.0)),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "width": int(w),
            "height": int(h),
            "channels": int(c),
            "aspect_ratio": round(aspect_ratio, 6),
            "total_pixels": int(w * h),
            "file_size_bytes": int(file_size),
            "file_path": str(path.resolve()),
            "analysis_width": int(aw),
            "analysis_height": int(ah),
            "analysis_max_dimension": int(analysis_max_dimension),
        },
        "analysis_method": {
            "photometric": "sRGB linear-light + perceptual LAB/HSV",
            "subject_method": subject_method,
            "subject_confidence": round(subject_confidence, 4),
            "external_masks_loaded": sorted(external_masks.keys()),
            "ai_requested": bool(enable_ai),
        },
        "exposure_and_tone": tone,
        "white_balance": wb,
        "contrast": contrast,
        "color_profile": color_profile,
        "quality": {
            **detail,
            # v2 compatibility alias; v3 value is estimated on flat regions.
            "noise_estimate_std": round(float(detail["noise_sigma_flat_regions"]["value"]), 4),
            "entropy": round(entropy, 4),
        },
        "scene": scene,
        "regions": {
            "semantic": semantic_regions,
            "grid_3x3": grid_regions,
        },
        "histograms": histograms,
        "ai": ai_result,
        "recommendations": recommendations,
        "classification": classification,
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data: Dict[str, Any], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid photographic image analyzer with confidence-aware recommendations."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_json", type=str, help="Path to save the JSON analysis.")
    parser.add_argument(
        "--analysis-size",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_DIMENSION,
        help=f"Maximum dimension used for analysis. Default: {DEFAULT_ANALYSIS_MAX_DIMENSION}.",
    )
    parser.add_argument(
        "--masks-json",
        type=str,
        default=None,
        help="Optional JSON containing paths to semantic mask images (person/face/skin/sky/etc.).",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Enable optional local zero-shot CLIP scene classification via transformers.",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Transformers model used by --ai.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a concise human-readable summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.analysis_size < 256:
        parser.error("--analysis-size must be at least 256.")

    try:
        data = extract_image_data(
            args.input_image,
            analysis_max_dimension=args.analysis_size,
            mask_json_path=args.masks_json,
            enable_ai=args.ai,
            ai_model=args.ai_model,
        )
        save_json(data, args.output_json)

        if args.pretty:
            rec = data["recommendations"]
            scene = data["scene"]
            wb = data["white_balance"]
            quality = data["quality"]
            classification = data["classification"]

            print("=" * 72)
            print("AUTO PHOTO EDITOR - HYBRID IMAGE ANALYSIS V3")
            print("=" * 72)
            print(f"Input              : {args.input_image}")
            print(f"Output             : {args.output_json}")
            print(f"Scene              : {scene['type']} ({scene['confidence']:.2f})")
            print(f"Exposure status    : {classification['exposure']['status']}")
            print(
                f"Exposure suggestion: {rec['exposure_ev']['value']:+.3f} EV "
                f"(confidence {rec['exposure_ev']['confidence']:.2f})"
            )
            print(
                f"WB cast            : temp {wb['estimated_temperature_cast']['value']:+.3f}, "
                f"tint {wb['estimated_tint_cast']['value']:+.3f} "
                f"(confidence {wb['estimated_temperature_cast']['confidence']:.2f})"
            )
            print(
                f"Noise sigma        : {quality['noise_sigma_flat_regions']['value']:.3f} "
                f"(confidence {quality['noise_sigma_flat_regions']['confidence']:.2f})"
            )
            print(f"Subject method     : {data['analysis_method']['subject_method']}")
            print(f"AI available       : {data['ai'].get('available', False)}")
            print("-" * 72)
            print("Suggested edits")
            print(f"  Exposure         : {rec['exposure_ev']['value']:+.3f} EV")
            print(f"  Highlights       : {rec['highlights']:+.1f}")
            print(f"  Shadows          : {rec['shadows']:+.1f}")
            print(f"  Contrast         : {rec['contrast']:+.1f}")
            print(f"  Temperature      : {rec['temperature']['value']:+.1f}")
            print(f"  Tint             : {rec['tint']['value']:+.1f}")
            print(f"  Noise reduction  : {rec['noise_reduction']['value']:+.1f}")
            print(f"  Sharpen          : {rec['sharpen']['value']:+.1f}")
            print("=" * 72)
        else:
            print(f"Saved hybrid image analysis to: {args.output_json}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Image analysis failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())