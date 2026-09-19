from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from autophotoeditor.core.image_io import RAW_EXTENSIONS, decode_color


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("masked_adaptive_enhance")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class GlobalConfig:
    # ------------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------------

    minimum_mask_pixels: int = 250
    minimum_effective_pixels: int = 80

    global_strength: float = 1.0

    # Mask edge
    feather_radius: float = 2.0

    # Context ring
    ring_inner_radius: int = 5
    ring_outer_radius: int = 35

    # ------------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------------

    analysis_max_dimension: int = 1400

    # ------------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------------

    exposure_dead_zone: float = 7.0
    exposure_match_strength: float = 0.45

    max_exposure_change: float = 28.0

    # Skin is deliberately conservative.
    skin_negative_exposure_factor: float = 0.28
    skin_positive_exposure_factor: float = 0.48

    # ------------------------------------------------------------------------
    # Tonal range
    # ------------------------------------------------------------------------

    shadow_strength: float = 0.30
    highlight_strength: float = 0.42

    max_shadow_lift: float = 18.0
    max_highlight_reduction: float = 24.0

    # ------------------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------------------

    contrast_strength: float = 0.22
    target_contrast: float = 58.0

    # ------------------------------------------------------------------------
    # Neutral color correction
    # ------------------------------------------------------------------------

    color_match_strength: float = 0.42

    max_a_change: float = 7.0
    max_b_change: float = 7.0

    # RGB neutral-gain protection.
    min_channel_gain: float = 0.94
    max_channel_gain: float = 1.06

    # ------------------------------------------------------------------------
    # Saturation / vibrance
    # ------------------------------------------------------------------------

    saturation_strength: float = 0.18

    # ------------------------------------------------------------------------
    # Detail
    # ------------------------------------------------------------------------

    clarity_strength: float = 0.18
    clarity_sigma: float = 10.0

    sharpening_strength: float = 0.65
    max_sharpen: float = 0.75

    # ------------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------------

    skin_color_protection: float = 0.65
    highlight_protection: float = 0.90

    # ------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------

    save_mask_results: bool = False


CONFIG = GlobalConfig()


# ============================================================================
# MASK PROFILE
# ============================================================================

@dataclass(frozen=True)
class MaskProfile:
    strength: float = 1.0

    # Feature amounts.
    wb: float = 1.0
    exposure: float = 1.0
    tone: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    clarity: float = 1.0
    sharpening: float = 1.0
    recovery: float = 1.0

    # Protection.
    saturation_limit: float = 1.0

    # Semantic category.
    category: str = "object"

    # Whether the region is skin.
    skin: bool = False

    # Reflective surfaces should not inherit surrounding colors.
    reflective: bool = False


# ============================================================================
# MASK PROFILES
# ============================================================================

MASK_PROFILES: Dict[str, MaskProfile] = {

    # ------------------------------------------------------------------------
    # Fine details
    # ------------------------------------------------------------------------

    "jewelry": MaskProfile(
        strength=0.52,
        wb=0.12,
        exposure=0.65,
        tone=0.75,
        contrast=0.90,
        saturation=0.35,
        clarity=0.85,
        sharpening=0.80,
        recovery=0.90,
        category="reflective",
        reflective=True,
    ),

    "glasses": MaskProfile(
        strength=0.48,
        wb=0.08,
        exposure=0.55,
        tone=0.70,
        contrast=0.80,
        saturation=0.25,
        clarity=0.70,
        sharpening=0.70,
        recovery=0.85,
        category="reflective",
        reflective=True,
    ),

    "eyes": MaskProfile(
        strength=0.55,
        wb=0.08,
        exposure=0.60,
        tone=0.70,
        contrast=0.85,
        saturation=0.30,
        clarity=0.75,
        sharpening=0.90,
        recovery=0.65,
        category="detail",
    ),

    "brows": MaskProfile(
        strength=0.48,
        wb=0.05,
        exposure=0.50,
        tone=0.65,
        contrast=0.75,
        saturation=0.15,
        clarity=0.65,
        sharpening=0.70,
        recovery=0.55,
        category="detail",
    ),

    "lips": MaskProfile(
        strength=0.46,
        wb=0.08,
        exposure=0.48,
        tone=0.65,
        contrast=0.65,
        saturation=0.45,
        clarity=0.35,
        sharpening=0.55,
        recovery=0.45,
        category="skin_detail",
        skin=True,
        saturation_limit=0.80,
    ),

    "mouth": MaskProfile(
        strength=0.38,
        wb=0.05,
        exposure=0.45,
        tone=0.60,
        contrast=0.55,
        saturation=0.25,
        clarity=0.25,
        sharpening=0.35,
        recovery=0.35,
        category="skin_detail",
        skin=True,
        saturation_limit=0.75,
    ),

    "nose": MaskProfile(
        strength=0.40,
        wb=0.08,
        exposure=0.48,
        tone=0.65,
        contrast=0.55,
        saturation=0.20,
        clarity=0.25,
        sharpening=0.35,
        recovery=0.40,
        category="skin_detail",
        skin=True,
        saturation_limit=0.78,
    ),

    "ears": MaskProfile(
        strength=0.40,
        wb=0.08,
        exposure=0.48,
        tone=0.65,
        contrast=0.50,
        saturation=0.20,
        clarity=0.20,
        sharpening=0.30,
        recovery=0.40,
        category="skin",
        skin=True,
        saturation_limit=0.80,
    ),

    # ------------------------------------------------------------------------
    # Skin
    # ------------------------------------------------------------------------

    "skin": MaskProfile(
        strength=0.50,
        wb=0.18,
        exposure=0.55,
        tone=0.68,
        contrast=0.35,
        saturation=0.20,
        clarity=0.12,
        sharpening=0.12,
        recovery=0.65,
        category="skin",
        skin=True,
        saturation_limit=0.72,
    ),

    "skin_face": MaskProfile(
        strength=0.48,
        wb=0.18,
        exposure=0.55,
        tone=0.68,
        contrast=0.35,
        saturation=0.20,
        clarity=0.10,
        sharpening=0.10,
        recovery=0.65,
        category="skin",
        skin=True,
        saturation_limit=0.72,
    ),

    "face": MaskProfile(
        strength=0.46,
        wb=0.18,
        exposure=0.52,
        tone=0.65,
        contrast=0.32,
        saturation=0.20,
        clarity=0.10,
        sharpening=0.10,
        recovery=0.62,
        category="skin",
        skin=True,
        saturation_limit=0.72,
    ),

    "neck": MaskProfile(
        strength=0.42,
        wb=0.18,
        exposure=0.50,
        tone=0.62,
        contrast=0.30,
        saturation=0.18,
        clarity=0.10,
        sharpening=0.10,
        recovery=0.60,
        category="skin",
        skin=True,
        saturation_limit=0.72,
    ),

    "hands": MaskProfile(
        strength=0.44,
        wb=0.16,
        exposure=0.52,
        tone=0.65,
        contrast=0.30,
        saturation=0.18,
        clarity=0.12,
        sharpening=0.12,
        recovery=0.58,
        category="skin",
        skin=True,
        saturation_limit=0.74,
    ),

    "arms": MaskProfile(
        strength=0.44,
        wb=0.16,
        exposure=0.52,
        tone=0.65,
        contrast=0.32,
        saturation=0.18,
        clarity=0.15,
        sharpening=0.14,
        recovery=0.58,
        category="skin",
        skin=True,
        saturation_limit=0.74,
    ),

    # ------------------------------------------------------------------------
    # Hair
    # ------------------------------------------------------------------------

    "hair": MaskProfile(
        strength=0.62,
        wb=0.28,
        exposure=0.72,
        tone=0.82,
        contrast=0.85,
        saturation=0.45,
        clarity=0.65,
        sharpening=0.62,
        recovery=0.72,
        category="hair",
    ),

    "hat": MaskProfile(
        strength=0.62,
        wb=0.25,
        exposure=0.70,
        tone=0.82,
        contrast=0.80,
        saturation=0.45,
        clarity=0.60,
        sharpening=0.55,
        recovery=0.70,
        category="object",
    ),

    # ------------------------------------------------------------------------
    # Clothing
    # ------------------------------------------------------------------------

    "top": MaskProfile(
        strength=0.72,
        wb=0.32,
        exposure=0.75,
        tone=0.86,
        contrast=0.80,
        saturation=0.65,
        clarity=0.60,
        sharpening=0.52,
        recovery=0.78,
        category="clothing",
    ),

    "dress": MaskProfile(
        strength=0.74,
        wb=0.32,
        exposure=0.76,
        tone=0.88,
        contrast=0.82,
        saturation=0.65,
        clarity=0.62,
        sharpening=0.55,
        recovery=0.78,
        category="clothing",
    ),

    "skirt": MaskProfile(
        strength=0.72,
        wb=0.32,
        exposure=0.74,
        tone=0.86,
        contrast=0.80,
        saturation=0.62,
        clarity=0.60,
        sharpening=0.52,
        recovery=0.76,
        category="clothing",
    ),

    "pants": MaskProfile(
        strength=0.72,
        wb=0.32,
        exposure=0.74,
        tone=0.86,
        contrast=0.80,
        saturation=0.60,
        clarity=0.58,
        sharpening=0.55,
        recovery=0.76,
        category="clothing",
    ),

    "clothing": MaskProfile(
        strength=0.68,
        wb=0.30,
        exposure=0.70,
        tone=0.82,
        contrast=0.75,
        saturation=0.58,
        clarity=0.55,
        sharpening=0.48,
        recovery=0.72,
        category="clothing",
    ),

    "belt": MaskProfile(
        strength=0.58,
        wb=0.22,
        exposure=0.62,
        tone=0.78,
        contrast=0.72,
        saturation=0.40,
        clarity=0.58,
        sharpening=0.52,
        recovery=0.72,
        category="object",
    ),

    "scarf": MaskProfile(
        strength=0.66,
        wb=0.30,
        exposure=0.68,
        tone=0.82,
        contrast=0.76,
        saturation=0.58,
        clarity=0.55,
        sharpening=0.48,
        recovery=0.72,
        category="clothing",
    ),

    "bag": MaskProfile(
        strength=0.64,
        wb=0.25,
        exposure=0.68,
        tone=0.80,
        contrast=0.78,
        saturation=0.48,
        clarity=0.58,
        sharpening=0.55,
        recovery=0.72,
        category="object",
    ),

    # ------------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------------

    "legs": MaskProfile(
        strength=0.48,
        wb=0.16,
        exposure=0.55,
        tone=0.68,
        contrast=0.38,
        saturation=0.20,
        clarity=0.22,
        sharpening=0.18,
        recovery=0.58,
        category="skin",
        skin=True,
        saturation_limit=0.74,
    ),

    "feet": MaskProfile(
        strength=0.44,
        wb=0.16,
        exposure=0.52,
        tone=0.65,
        contrast=0.35,
        saturation=0.18,
        clarity=0.18,
        sharpening=0.15,
        recovery=0.55,
        category="skin",
        skin=True,
        saturation_limit=0.74,
    ),

    "torso": MaskProfile(
        strength=0.55,
        wb=0.22,
        exposure=0.62,
        tone=0.72,
        contrast=0.48,
        saturation=0.28,
        clarity=0.32,
        sharpening=0.28,
        recovery=0.62,
        category="body",
    ),

    "body": MaskProfile(
        strength=0.46,
        wb=0.18,
        exposure=0.55,
        tone=0.66,
        contrast=0.38,
        saturation=0.20,
        clarity=0.25,
        sharpening=0.20,
        recovery=0.55,
        category="body",
    ),

    "person": MaskProfile(
        strength=0.32,
        wb=0.12,
        exposure=0.45,
        tone=0.55,
        contrast=0.25,
        saturation=0.15,
        clarity=0.18,
        sharpening=0.12,
        recovery=0.45,
        category="person",
    ),

    # ------------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------------

    "background": MaskProfile(
        strength=0.62,
        wb=0.30,
        exposure=0.65,
        tone=0.82,
        contrast=0.75,
        saturation=0.55,
        clarity=0.55,
        sharpening=0.45,
        recovery=0.78,
        category="background",
    ),
}


DEFAULT_PROFILE = MaskProfile()


# ============================================================================
# PRIORITY
# ============================================================================

MASK_PRIORITY = [
    "jewelry",
    "glasses",
    "eyes",
    "brows",
    "lips",
    "mouth",
    "nose",
    "ears",

    "skin_face",
    "skin",
    "face",
    "neck",
    "hands",
    "arms",
    "legs",
    "feet",

    "hat",
    "hair",

    "bag",
    "belt",
    "scarf",

    "top",
    "dress",
    "skirt",
    "pants",
    "clothing",

    "torso",
    "body",
    "person",

    "background",
]


# ============================================================================
# IMAGE I/O
# ============================================================================

