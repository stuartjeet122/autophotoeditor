#!/usr/bin/env python3
"""
apply_preset.py

Lightroom XMP-inspired image processor.

Usage examples
--------------

Apply ALL supported XMP settings:
    python apply_preset.py preset.xmp input.jpg output.jpg

Apply only HSL:
    python apply_preset.py preset.xmp input.jpg output.jpg --hsl

Apply only tone:
    python apply_preset.py preset.xmp input.jpg output.jpg --tone

Apply HSL + effects:
    python apply_preset.py preset.xmp input.jpg output.jpg --hsl --effects

Override individual tone values:
    python apply_preset.py preset.xmp input.jpg output.jpg --contrast -20

Override HSL:
    python apply_preset.py preset.xmp input.jpg output.jpg \
        --hsl-red "-10,-20,10"

Format for HSL:
    hue,saturation,luminance

Example:
    --hsl-red "-10,-20,10"

If NO processing selector is supplied:
    ALL supported XMP settings are applied.

If at least one processing selector is supplied:
    ONLY selected categories are applied.

Individual parameter arguments automatically enable their category.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autophotoeditor.core.image_io import RAW_EXTENSIONS, decode_color


# ============================================================================
# CONSTANTS
# ============================================================================

HSL_CHANNELS = (
    "red",
    "orange",
    "yellow",
    "green",
    "aqua",
    "blue",
    "purple",
    "magenta",
)

# Approximate hue centers in degrees.
HSL_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 315.0,
}

EPS = 1e-8


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HSLAdjustment:
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0


@dataclass
class ToneSettings:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0


@dataclass
class ColorSettings:
    vibrance: float = 0.0
    saturation: float = 0.0


@dataclass
class WhiteBalanceSettings:
    temperature: float = 0.0
    tint: float = 0.0


@dataclass
class EffectsSettings:
    clarity: float = 0.0
    texture: float = 0.0
    dehaze: float = 0.0


@dataclass
class SharpenSettings:
    amount: float = 0.0
    radius: float = 1.0
    detail: float = 25.0
    masking: float = 0.0


@dataclass
class NoiseSettings:
    luminance: float = 0.0
    luminance_detail: float = 50.0
    color: float = 0.0


@dataclass
class PointCurve:
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class XMPSettings:
    tone: ToneSettings = field(default_factory=ToneSettings)
    wb: WhiteBalanceSettings = field(
        default_factory=WhiteBalanceSettings
    )
    color: ColorSettings = field(default_factory=ColorSettings)
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    sharpen: SharpenSettings = field(default_factory=SharpenSettings)
    noise: NoiseSettings = field(default_factory=NoiseSettings)

    hsl: Dict[str, HSLAdjustment] = field(
        default_factory=lambda: {
            channel: HSLAdjustment()
            for channel in HSL_CHANNELS
        }
    )

    point_curve: PointCurve = field(default_factory=PointCurve)

    camera_profile: Optional[str] = None
    process_version: Optional[str] = None

    tone_curve_name: Optional[str] = None
    tone_curve_name_2012: Optional[str] = None

    unsupported: List[str] = field(default_factory=list)


@dataclass
class PipelineOptions:
    use_exposure: bool = False
    use_white_balance: bool = False
    use_tone: bool = False
    use_point_curve: bool = False
    use_hsl: bool = False
    use_color: bool = False
    use_effects: bool = False
    use_noise: bool = False
    use_sharpening: bool = False

    debug: bool = False


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def finite_image(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def normalize_angle_degrees(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 360.0)


def angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """
    Circular distance between hue arrays and hue center.
    Result is 0..180 degrees.
    """
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return np.zeros_like(x)

    t = np.clip(
        (x - edge0) / (edge1 - edge0),
        0.0,
        1.0,
    )

    return t * t * (3.0 - 2.0 * t)


