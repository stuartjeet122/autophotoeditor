"""
AutoPhotoEditor - Image Analysis V4
===================================

Purpose
-------
High-quality image analysis for automatic photo enhancement.

Design goals
------------
1. Lightroom-like adaptive Auto Tone behaviour.
2. Linear-light exposure analysis.
3. Subject-aware luminance optimization.
4. Real neutral white-balance estimation.
5. Multiple WB estimators + agreement validation.
6. Semantic-mask awareness.
7. Fast deterministic processing.
8. Backwards-compatible JSON sections.

Input
-----
BGR uint8 image.

Output
------
JSON-compatible dictionary.

Important
---------
This analyzer does NOT itself modify the image.
It produces an edit/correction plan for the enhancer.

Recommended OpenCV package
---------------------------
opencv-contrib-python

Optional:
    pip install colour-science

The analyzer does not require colour-science to run.
"""

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

logger = logging.getLogger("image_analysis_v4")


# ============================================================
# CONSTANTS
# ============================================================

SCHEMA_VERSION = "4.0"

EPS = 1e-8

MAX_ANALYSIS_DIM = 1600
MAX_OPTIMIZATION_PIXELS = 250_000
MAX_WB_PIXELS = 180_000

DEFAULT_STRENGTH = 1.0

# OpenCV 8-bit LAB:
# L = 0..255  -> CIE L* approximately 0..100
# a,b = 0..255 with 128 approximately neutral.
LAB_L_SCALE = 100.0 / 255.0

# Reasonable physical RGB gain limits.
MIN_WB_GAIN = 0.65
MAX_WB_GAIN = 1.55



from .shared import *


def _decode_rle(
    rle: Sequence[int],
    height: int,
    width: int,
) -> Optional[np.ndarray]:

    try:
        flat = np.zeros(height * width, dtype=np.uint8)

        index = 0
        value = 0

        for count in rle:
            count = int(count)

            if count < 0:
                return None

            end = min(index + count, flat.size)

            if value:
                flat[index:end] = 1

            index = end
            value = 1 - value

            if index >= flat.size:
                break

        return flat.reshape(height, width).astype(bool)

    except Exception:
        return None


def decode_external_mask(
    value: Any,
    height: int,
    width: int,
) -> Optional[np.ndarray]:

    if value is None:
        return None

    if isinstance(value, dict):

        if "rle" in value:
            rle = value.get("rle")

            if isinstance(rle, list):
                h = safe_int(
                    value.get("height"),
                    height,
                )

                w = safe_int(
                    value.get("width"),
                    width,
                )

                mask = _decode_rle(
                    rle,
                    h,
                    w,
                )

                if mask is not None:
                    return resize_mask(
                        mask,
                        (height, width),
                    )

        for key in (
            "mask",
            "data",
            "array",
        ):
            if key in value:
                return decode_external_mask(
                    value[key],
                    height,
                    width,
                )

        return None

    if isinstance(value, list):

        try:
            arr = np.asarray(value)

            if arr.ndim == 2:
                return resize_mask(
                    arr > 0,
                    (height, width),
                )

        except Exception:
            return None

    return None


def load_masks(
    masks_json: Optional[Any],
    shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:

    h, w = shape

    if masks_json is None:
        return {}

    data: Any = masks_json

    if isinstance(masks_json, (str, Path)):

        path = Path(masks_json)

        if not path.exists():
            logger.warning(
                "Mask JSON does not exist: %s",
                path,
            )
            return {}

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to read masks JSON: %s",
                exc,
            )
            return {}

    if not isinstance(data, Mapping):
        return {}

    source = data.get("masks", data)

    if not isinstance(source, Mapping):
        return {}

    result: Dict[str, np.ndarray] = {}

    for name, value in source.items():

        mask = decode_external_mask(
            value,
            h,
            w,
        )

        if mask is not None:
            result[str(name).lower()] = mask

    return result


# ============================================================
# SALIENCY
# ============================================================

def spectral_residual_saliency(
    image: np.ndarray,
) -> np.ndarray:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    gray = np.maximum(
        gray,
        1.0,
    )

    fft = np.fft.fft2(gray)

    magnitude = np.abs(fft)

    phase = np.angle(fft)

    log_amplitude = np.log(
        magnitude + 1e-6,
    )

    avg = cv2.blur(
        log_amplitude.astype(np.float32),
        (3, 3),
    )

    spectral_residual = log_amplitude - avg

    saliency = np.abs(
        np.fft.ifft2(
            np.exp(
                spectral_residual + 1j * phase
            )
        )
    ) ** 2

    saliency = cv2.GaussianBlur(
        saliency.astype(np.float32),
        (9, 9),
        0,
    )

    mn = float(np.min(saliency))
    mx = float(np.max(saliency))

    if mx - mn < EPS:
        return np.zeros_like(saliency)

    saliency = (
        saliency - mn
    ) / (
        mx - mn
    )

    return saliency.astype(np.float32)


def derive_subject_mask(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:

    shape = image.shape[:2]

    for key in (
        "subject",
        "person",
        "foreground",
        "main_subject",
    ):

        if key in masks:
            return (
                resize_mask(
                    masks[key],
                    shape,
                ),
                f"external:{key}",
            )

    semantic = union_masks(
        masks.get("person"),
        masks.get("skin"),
        masks.get("face"),
        masks.get("hair"),
        masks.get("eyes"),
        shape=shape,
    )

    if semantic is not None:

        coverage = mask_fraction(
            semantic
        )

        if 0.01 <= coverage <= 0.95:
            return (
                semantic,
                "external:semantic",
            )

    saliency = spectral_residual_saliency(
        image
    )

    h, w = shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    cx = w * 0.5
    cy = h * 0.5

    dx = (xx - cx) / max(w, 1)
    dy = (yy - cy) / max(h, 1)

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    center_prior = np.exp(
        -(distance ** 2) / 0.18
    )

    combined = (
        0.72 * saliency
        + 0.28 * center_prior
    )

    threshold = float(
        np.percentile(
            combined,
            82,
        )
    )

    subject = combined >= threshold

    kernel = np.ones(
        (9, 9),
        np.uint8,
    )

    subject = cv2.morphologyEx(
        subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )

    subject = cv2.morphologyEx(
        subject,
        cv2.MORPH_OPEN,
        kernel,
    )

    coverage = float(
        np.mean(subject)
    )

    if coverage < 0.02 or coverage > 0.75:
        return None, "none"

    return (
        subject.astype(bool),
        "saliency",
    )


# ============================================================
# SEMANTIC HEURISTICS
# ============================================================

def derive_semantic_masks(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:

    h, w = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    bgr = image.astype(np.float32) / 255.0

    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    saturation = hsv[:, :, 1].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)

    # ----------------------------
    # Skin
    # ----------------------------

    skin_external = union_masks(
        masks.get("skin"),
        masks.get("face"),
        shape=(h, w),
    )

    skin_heuristic = (
        (r > g * 1.03)
        & (g > b * 1.02)
        & (r > 0.20)
        & (g > 0.12)
        & (saturation > 20)
        & (saturation < 190)
        & (hue < 35)
    )

    skin = union_masks(
        skin_external,
        skin_heuristic,
        shape=(h, w),
    )

    # ----------------------------
    # Vegetation
    # ----------------------------

    vegetation_external = masks.get(
        "vegetation",
        masks.get("green"),
    )

    vegetation_heuristic = (
        (g > r * 1.05)
        & (g > b * 1.05)
        & (hue >= 30)
        & (hue <= 95)
        & (saturation > 35)
    )

    vegetation = union_masks(
        vegetation_external,
        vegetation_heuristic,
        shape=(h, w),
    )

    # ----------------------------
    # Sky
    # ----------------------------

    sky_external = masks.get("sky")

    sky_heuristic = (
        (b > r * 1.03)
        & (b > g * 0.96)
        & (hue >= 85)
        & (hue <= 135)
        & (saturation > 25)
    )

    sky = union_masks(
        sky_external,
        sky_heuristic,
        shape=(h, w),
    )

    return {
        "skin": skin,
        "vegetation": vegetation,
        "sky": sky,
    }


