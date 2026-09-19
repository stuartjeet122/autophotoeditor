from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from autophotoeditor.core.image_io import decode_color


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("mask_auto_enhancement")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def fail(message: str) -> None:
    logger.error(message)
    raise SystemExit(1)


def choose_device(requested: str = "auto") -> str:
    """Resolve the API device setting to a device supported by this host."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")

    if normalized == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8

# Lightroom-like limits.
# These are deliberately conservative.
LIMITS = {
    "exposure": (-1.50, 1.50),
    "contrast": (-35.0, 35.0),
    "highlights": (-60.0, 60.0),
    "shadows": (-60.0, 60.0),
    "whites": (-35.0, 35.0),
    "blacks": (-35.0, 35.0),
    "temperature": (-20.0, 20.0),
    "tint": (-15.0, 15.0),
    "saturation": (-25.0, 25.0),
    "vibrance": (-30.0, 30.0),
    "clarity": (-20.0, 20.0),
    "dehaze": (-15.0, 15.0),
}

# Mask-specific aggressiveness.
MASK_STRENGTH = {
    "person": 0.45,

    "face": 0.50,
    "skin": 0.38,
    "skin_face": 0.35,
    "skin_neck": 0.35,

    "eyes": 0.25,
    "brows": 0.20,
    "ears": 0.30,
    "nose": 0.30,
    "mouth": 0.30,
    "lips": 0.28,

    "hair": 0.35,

    "clothing": 0.40,
    "top": 0.40,
    "dress": 0.40,
    "skirt": 0.40,
    "pants": 0.40,
    "belt": 0.30,

    "arms": 0.35,
    "hands": 0.32,
    "legs": 0.35,
    "feet": 0.30,
    "torso": 0.38,

    "glasses": 0.20,
    "hat": 0.30,
    "scarf": 0.35,
    "bag": 0.35,
    "jewelry": 0.20,

    "neck": 0.32,
    "body": 0.35,
}

# Very small masks should not receive strong corrections.
MIN_PIXELS = {
    "eyes": 20,
    "brows": 15,
    "nose": 20,
    "mouth": 20,
    "lips": 15,
    "jewelry": 10,
    "glasses": 20,
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def clip(
    value: float,
    low: float,
    high: float,
) -> float:
    return float(np.clip(value, low, high))


def robust_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    if values.size == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def weighted_percentile(
    values: np.ndarray,
    weights: Optional[np.ndarray],
    percentile: float,
) -> float:
    """
    Weighted percentile.

    Used for soft segmentation masks.
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

    valid = (
        np.isfinite(values)
        &
        np.isfinite(weights)
        &
        (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return 0.0

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    target = (
        percentile / 100.0
    ) * cumulative[-1]

    index = np.searchsorted(
        cumulative,
        target,
        side="left",
    )

    index = min(
        max(index, 0),
        len(values) - 1,
    )

    return float(values[index])


def robust_mean(
    values: np.ndarray,
    low: float = 5.0,
    high: float = 95.0,
) -> float:

    if values.size == 0:
        return 0.0

    p1 = np.percentile(
        values,
        low,
    )

    p2 = np.percentile(
        values,
        high,
    )

    selected = values[
        (values >= p1)
        &
        (values <= p2)
    ]

    if selected.size == 0:
        return float(np.mean(values))

    return float(
        np.mean(selected)
    )


def safe_ratio(
    a: float,
    b: float,
) -> float:
    return float(
        a / max(abs(b), EPS)
    )


# ============================================================
# COLOR SCIENCE
# ============================================================

def srgb_to_linear(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized sRGB to linear RGB.
    """

    x = np.clip(
        image.astype(np.float32),
        0.0,
        1.0,
    )

    return np.where(
        x <= 0.04045,
        x / 12.92,
        (
            (x + 0.055) / 1.055
        ) ** 2.4,
    )


def linear_to_srgb(
    image: np.ndarray,
) -> np.ndarray:

    x = np.clip(
        image.astype(np.float32),
        0.0,
        1.0,
    )

    return np.where(
        x <= 0.0031308,
        x * 12.92,
        (
            1.055
            *
            np.power(
                x,
                1.0 / 2.4,
            )
            -
            0.055
        ),
    )


def bgr_to_linear_rgb(
    image_bgr: np.ndarray,
) -> np.ndarray:

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    rgb = (
        rgb.astype(np.float32)
        / 255.0
    )

    return srgb_to_linear(
        rgb
    )


def luminance_from_linear_rgb(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    CIE / Rec.709 luminance.
    """

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    )


def linear_rgb_to_lab(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert linear RGB to CIE Lab.

    D65 white point.
    """

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

    # D65.
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    delta = 6.0 / 29.0

    def f(v: np.ndarray) -> np.ndarray:
        return np.where(
            v > delta ** 3,
            np.cbrt(
                np.maximum(v, 0)
            ),
            v / (
                3 * delta * delta
            )
            +
            4.0 / 29.0,
        )

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
        [L, a, b_value],
        axis=-1,
    )


# ============================================================
# IMAGE ANALYSIS CACHE
# ============================================================

@dataclass
class ImageColorData:
    linear_rgb: np.ndarray
    luminance: np.ndarray
    lab: np.ndarray
    hsv: np.ndarray
    bgr_float: np.ndarray


def prepare_color_data(
    image: np.ndarray,
) -> ImageColorData:

    bgr_float = (
        image.astype(np.float32)
        / 255.0
    )

    linear_rgb = bgr_to_linear_rgb(
        image
    )

    luminance = luminance_from_linear_rgb(
        linear_rgb
    )

    lab = linear_rgb_to_lab(
        linear_rgb
    )

    hsv = cv2.cvtColor(
        bgr_float,
        cv2.COLOR_BGR2HSV,
    )

    return ImageColorData(
        linear_rgb=linear_rgb,
        luminance=luminance,
        lab=lab,
        hsv=hsv,
        bgr_float=bgr_float,
    )


