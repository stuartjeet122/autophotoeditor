#!/usr/bin/env python3

"""
lens_correction.py

Lens correction engine for AutoPhotoEditor.

Features
--------

1. Automatic EXIF camera/lens detection
2. Lensfun database lookup
3. Lensfun distortion correction
4. Lensfun TCA correction
5. Lensfun vignetting correction
6. Manual OpenCV distortion correction
7. Manual distortion strength
8. Manual optical center
9. Manual focal scaling
10. Automatic crop
11. Keep-full-frame mode
12. Profile listing/search
13. Debug grid
14. JPEG / PNG / TIFF support

Examples
--------

Automatic correction from EXIF:

    python lens_correction.py input.jpg output.jpg


Specify camera/lens manually:

    python lens_correction.py input.jpg output.jpg \
        --camera "Canon EOS 5D Mark IV" \
        --camera-maker "Canon" \
        --lens-maker "Sigma" \
        --lens "Sigma 24-70mm F2.8 DG OS HSM"


Manual correction:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --k3 0.0


Manual correction strength:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --strength 0.75


Manual center:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --center-x 0.50 \
        --center-y 0.50


Search database:

    python lens_correction.py \
        --search-lens "24-70"


List cameras:

    python lens_correction.py \
        --list-cameras "Canon"


List lenses:

    python lens_correction.py \
        --list-lenses "Sigma"


Debug:

    python lens_correction.py input.jpg output.jpg \
        --debug
"""


from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from autophotoeditor.core.image_io import RAW_EXTENSIONS, decode_color


# ============================================================================
# OPTIONAL LENSPY IMPORT
# ============================================================================

try:
    import lensfunpy
except ImportError:
    lensfunpy = None


# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-8


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImageMetadata:
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_make: Optional[str] = None
    lens_model: Optional[str] = None

    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    focus_distance: Optional[float] = None

    width: int = 0
    height: int = 0


@dataclass
class ManualDistortion:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    p1: float = 0.0
    p2: float = 0.0

    center_x: float = 0.5
    center_y: float = 0.5

    focal_scale: float = 1.0

    strength: float = 1.0


@dataclass
class LensCorrectionOptions:
    use_lensfun: bool = True

    distortion: bool = True
    tca: bool = False
    vignetting: bool = False

    manual: bool = False

    auto_crop: bool = True
    crop_factor: float = 1.0

    interpolation: int = cv2.INTER_LANCZOS4

    debug: bool = False
    debug_grid: bool = False


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def normalize_string(
    value: Optional[str],
) -> Optional[str]:

    if value is None:
        return None

    value = str(value)

    value = (
        value
        .replace("\x00", "")
        .strip()
    )

    if not value:
        return None

    return value


def finite_image(
    image: np.ndarray,
) -> np.ndarray:

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


