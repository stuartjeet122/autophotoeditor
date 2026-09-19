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

def imread_unicode(
    path: Path,
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:

    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            image = decode_color(path)

            if flags == cv2.IMREAD_GRAYSCALE:
                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

            return image

        except Exception as exc:
            logger.warning(
                "Could not decode RAW image %s: %s",
                path,
                exc,
            )
            return None

    try:
        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        if data.size == 0:
            return None

        return cv2.imdecode(
            data,
            flags,
        )

    except Exception as exc:
        logger.warning(
            "Could not read image %s: %s",
            path,
            exc,
        )
        return None


def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    quality: int = 95,
) -> bool:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(quality),
        ]

    elif suffix == ".png":
        ext = ".png"
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    elif suffix == ".webp":
        ext = ".webp"
        params = [
            cv2.IMWRITE_WEBP_QUALITY,
            int(quality),
        ]

    elif suffix in {".tif", ".tiff"}:
        ext = suffix
        params = [
            cv2.IMWRITE_TIFF_COMPRESSION,
            1,
        ]

    else:
        logger.error(
            "Unsupported output extension: %s",
            suffix,
        )
        return False

    try:
        ok, encoded = cv2.imencode(
            ext,
            image,
            params,
        )

        if not ok:
            return False

        encoded.tofile(str(path))
        return True

    except Exception as exc:
        logger.error(
            "Could not save %s: %s",
            path,
            exc,
        )
        return False


# ============================================================================
# PATH / JSON HELPERS
# ============================================================================

MASK_FILE_KEYS = {
    "file",
    "path",
    "mask_file",
    "mask_path",
    "png",
    "filename",
    "mask_filename",
    "soft_file",
    "binary_file",
}

MASK_BINARY_KEYS = {
    "binary_png_base64",
    "binary_base64",
    "png_base64",
    "mask_png_base64",
}


def resolve_path(
    raw_path: str,
    json_path: Path,
    mask_dir: Optional[Path],
) -> Optional[Path]:

    if not raw_path:
        return None

    raw = Path(str(raw_path))
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    else:

        if mask_dir is not None:
            candidates.append(
                mask_dir / raw
            )

        candidates.append(
            json_path.parent / raw
        )

        if mask_dir is not None:
            candidates.append(
                mask_dir / raw.name
            )

        candidates.append(
            json_path.parent / raw.name
        )

    for candidate in candidates:

        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            pass

    return None


def extract_mask_entries(
    data: Any,
    json_path: Path,
    mask_dir: Optional[Path],
) -> Dict[str, Path]:

    result: Dict[str, Path] = {}
    embedded_dir = Path(tempfile.mkdtemp(prefix="autophotoeditor_masks_"))

    def add_entry(
        name: str,
        value: Any,
    ) -> None:

        name = str(name).strip().lower()

        if not name:
            return

        if name in {
            "image",
            "input",
            "source",
            "original",
            "photo",
        }:
            return

        path: Optional[Path] = None

        if isinstance(value, str):
            path = resolve_path(
                value,
                json_path,
                mask_dir,
            )

        elif isinstance(value, dict):

            for key in MASK_BINARY_KEYS:

                encoded = value.get(key)

                if not isinstance(encoded, str):
                    continue

                try:
                    binary = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    continue

                path = embedded_dir / f"{name}.png"
                path.write_bytes(binary)
                break

            if path is None:

                for key in MASK_FILE_KEYS:

                    raw = value.get(key)

                    if not isinstance(raw, str):
                        continue

                    path = resolve_path(
                        raw,
                        json_path,
                        mask_dir,
                    )

                    if path is not None:
                        break

        if path is None:
            return

        if path.suffix.lower() != ".png":
            return

        result[name] = path

    if not isinstance(data, dict):
        return result

    masks = data.get("masks")

    if isinstance(masks, dict):

        for name, value in masks.items():
            add_entry(
                name,
                value,
            )

    elif isinstance(masks, list):

        for item in masks:

            if not isinstance(item, dict):
                continue

            name = (
                item.get("name")
                or item.get("label")
                or item.get("class")
                or item.get("mask")
            )

            if name:
                add_entry(
                    str(name),
                    item,
                )

    mask_files = data.get("mask_files")

    if isinstance(mask_files, dict):

        for name, value in mask_files.items():
            add_entry(
                name,
                value,
            )

    elif isinstance(mask_files, list):

        for item in mask_files:

            if isinstance(item, dict):

                name = (
                    item.get("name")
                    or item.get("label")
                    or item.get("class")
                )

                if name:
                    add_entry(
                        str(name),
                        item,
                    )

            elif isinstance(item, str):

                path = resolve_path(
                    item,
                    json_path,
                    mask_dir,
                )

                if (
                    path is not None
                    and path.suffix.lower() == ".png"
                ):
                    result.setdefault(
                        path.stem.lower(),
                        path,
                    )

    return result


def discover_masks_from_directory(
    mask_dir: Path,
) -> Dict[str, Path]:

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}"
        )

    result: Dict[str, Path] = {}

    for path in mask_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() != ".png":
            continue

        name = path.stem.lower()

        if name.startswith(
            (
                "preview",
                "visual",
                "debug",
            )
        ):
            continue

        result[name] = path.resolve()

    return result


# ============================================================================
# MASK LOADING
# ============================================================================

def load_mask(
    path: Path,
    width: int,
    height: int,
) -> Optional[np.ndarray]:

    mask = imread_unicode(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        logger.warning(
            "Could not load mask: %s",
            path,
        )
        return None

    if mask.ndim != 2:
        return None

    mh, mw = mask.shape

    if mw != width or mh != height:
        logger.info(
            "Resizing mask: %s (mask=%dx%d image=%dx%d)",
            path,
            mw,
            mh,
            width,
            height,
        )
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    result = (
        mask.astype(np.float32)
        / 255.0
    )

    result[result < 0.01] = 0.0

    return result


def clean_mask(
    mask: np.ndarray,
) -> np.ndarray:

    mask_u8 = np.clip(
        mask * 255.0,
        0,
        255,
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return (
        mask_u8.astype(np.float32)
        / 255.0
    )


def feather_mask(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:

    if radius <= 0:
        return mask.astype(
            np.float32,
            copy=True,
        )

    sigma = max(
        radius / 2.0,
        0.5,
    )

    result = cv2.GaussianBlur(
        mask.astype(np.float32),
        (0, 0),
        sigma,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================================
# IMAGE ANALYSIS CACHE
# ============================================================================

class ImageAnalysis:

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:

        self.image = image

        self.height, self.width = image.shape[:2]

        self.gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        self.luma = self.gray / 255.0

        self.lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        self.lab_a = (
            self.lab[:, :, 1]
            .astype(np.float32)
        )

        self.lab_b = (
            self.lab[:, :, 2]
            .astype(np.float32)
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        self.saturation = (
            hsv[:, :, 1]
            .astype(np.float32)
            / 255.0
        )

        self.value = (
            hsv[:, :, 2]
            .astype(np.float32)
            / 255.0
        )

        # Full-image Laplacian is calculated only once.
        self.laplacian = cv2.Laplacian(
            self.gray,
            cv2.CV_32F,
        )

    def crop(
        self,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
    ) -> "ImageAnalysis":
        """Create a lightweight view over already-computed analysis arrays."""

        result = object.__new__(ImageAnalysis)
        result.image = self.image[y1:y2, x1:x2]
        result.height = y2 - y1
        result.width = x2 - x1
        result.gray = self.gray[y1:y2, x1:x2]
        result.luma = self.luma[y1:y2, x1:x2]
        result.lab = self.lab[y1:y2, x1:x2]
        result.lab_a = self.lab_a[y1:y2, x1:x2]
        result.lab_b = self.lab_b[y1:y2, x1:x2]
        result.saturation = self.saturation[y1:y2, x1:x2]
        result.value = self.value[y1:y2, x1:x2]
        result.laplacian = self.laplacian[y1:y2, x1:x2]
        return result


# ============================================================================
# STATISTICS
# ============================================================================

def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:

    if values.size == 0:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not np.any(valid):
        return float(
            np.percentile(
                values,
                percentile,
            )
        )

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    target = (
        cumulative[-1]
        * percentile
        / 100.0
    )

    index = int(
        np.searchsorted(
            cumulative,
            target,
        )
    )

    index = max(
        0,
        min(
            index,
            len(values) - 1,
        ),
    )

    return float(values[index])


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:

    if values.size == 0:
        return 0.0

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    total = float(
        np.sum(weights)
    )

    if total <= 1e-8:
        return float(
            np.mean(values)
        )

    return float(
        np.sum(values * weights)
        / total
    )


def analyze_region(
    analysis: ImageAnalysis,
    mask: np.ndarray,
) -> Dict[str, float]:

    valid = mask > 0.05

    if not np.any(valid):
        return {
            "pixels": 0,
            "brightness": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "saturation": 0.0,
            "shadow_fraction": 0.0,
            "highlight_fraction": 0.0,
            "clipped_fraction": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "lab_a": 128.0,
            "lab_b": 128.0,
        }

    weights = mask[valid]

    luma = analysis.luma[valid]

    saturation = analysis.saturation[valid]

    lab_a = analysis.lab_a[valid]
    lab_b = analysis.lab_b[valid]

    lap = analysis.laplacian[valid]

    p10 = weighted_percentile(
        luma,
        weights,
        10,
    )

    p25 = weighted_percentile(
        luma,
        weights,
        25,
    )

    p50 = weighted_percentile(
        luma,
        weights,
        50,
    )

    p75 = weighted_percentile(
        luma,
        weights,
        75,
    )

    p90 = weighted_percentile(
        luma,
        weights,
        90,
    )

    return {
        "pixels": int(
            np.count_nonzero(valid)
        ),

        "brightness": weighted_mean(
            luma,
            weights,
        ),

        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,

        "saturation": weighted_mean(
            saturation,
            weights,
        ),

        "shadow_fraction": float(
            np.mean(luma < 0.10)
        ),

        "highlight_fraction": float(
            np.mean(luma > 0.92)
        ),

        "clipped_fraction": float(
            np.mean(luma > 0.985)
        ),

        "contrast": float(
            p90 - p10
        ),

        "sharpness": float(
            np.var(lap)
        ),

        "lab_a": weighted_mean(
            lab_a,
            weights,
        ),

        "lab_b": weighted_mean(
            lab_b,
            weights,
        ),
    }


# ============================================================================
# BOUNDING BOX
# ============================================================================

def mask_bbox(
    mask: np.ndarray,
    padding: int,
) -> Optional[Tuple[int, int, int, int]]:

    ys, xs = np.where(
        mask > 0.02
    )

    if len(xs) == 0:
        return None

    h, w = mask.shape

    x1 = max(
        0,
        int(xs.min()) - padding,
    )

    y1 = max(
        0,
        int(ys.min()) - padding,
    )

    x2 = min(
        w,
        int(xs.max()) + padding + 1,
    )

    y2 = min(
        h,
        int(ys.max()) + padding + 1,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================================
# CONTEXT / SURROUNDING REGION
# ============================================================================

def surrounding_ring(
    mask: np.ndarray,
    all_masks: Optional[np.ndarray] = None,
    inner_radius: int = 5,
    outer_radius: int = 35,
) -> np.ndarray:

    mask_u8 = np.where(
        mask > 0.05,
        255,
        0,
    ).astype(np.uint8)

    if outer_radius <= 0:
        return np.zeros_like(
            mask,
            dtype=np.float32,
        )

    outer_kernel_size = (
        outer_radius * 2 + 1
    )

    inner_kernel_size = (
        inner_radius * 2 + 1
    )

    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            outer_kernel_size,
            outer_kernel_size,
        ),
    )

    outer = cv2.dilate(
        mask_u8,
        outer_kernel,
    )

    if inner_radius > 0:

        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                inner_kernel_size,
                inner_kernel_size,
            ),
        )

        inner = cv2.dilate(
            mask_u8,
            inner_kernel,
        )

    else:
        inner = mask_u8

    ring = (
        outer.astype(np.float32)
        -
        inner.astype(np.float32)
    )

    ring = np.clip(
        ring / 255.0,
        0.0,
        1.0,
    )

    # Exclude other semantic masks.
    if all_masks is not None:
        ring[
            all_masks > 0.05
        ] = 0.0

    return ring


def analyze_surrounding(
    analysis: ImageAnalysis,
    mask: np.ndarray,
    all_masks: Optional[np.ndarray] = None,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
]:

    ring = surrounding_ring(
        mask,
        all_masks=all_masks,
        inner_radius=CONFIG.ring_inner_radius,
        outer_radius=CONFIG.ring_outer_radius,
    )

    if np.count_nonzero(ring > 0.05) < 50:

        # Fallback to a slightly wider ring.
        ring = surrounding_ring(
            mask,
            all_masks=None,
            inner_radius=3,
            outer_radius=55,
        )

    stats = analyze_region(
        analysis,
        ring,
    )

    return stats, ring


# ============================================================================
# GLOBAL NEUTRAL ILLUMINANT
# ============================================================================

def estimate_global_neutral_gains(
    analysis: ImageAnalysis,
) -> Tuple[float, float, float]:

    image = analysis.image

    # Work on a reduced image for speed.
    h, w = image.shape[:2]

    scale = min(
        1.0,
        900.0 / max(h, w),
    )

    if scale < 1.0:

        small = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:
        small = image

    b = small[:, :, 0].astype(
        np.float32
    )

    g = small[:, :, 1].astype(
        np.float32
    )

    r = small[:, :, 2].astype(
        np.float32
    )

    gray = (
        0.114 * b
        + 0.587 * g
        + 0.299 * r
    )

    mx = np.maximum.reduce(
        [r, g, b]
    )

    mn = np.minimum.reduce(
        [r, g, b]
    )

    chroma_ratio = (
        (mx - mn)
        / np.maximum(mx, 1.0)
    )

    # Neutral-ish, non-extreme pixels.
    valid = (
        (gray > 30)
        & (gray < 235)
        & (chroma_ratio < 0.18)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 100:

        return (
            1.0,
            1.0,
            1.0,
        )

    rv = r[valid]
    gv = g[valid]
    bv = b[valid]

    # Median channel values are more robust than a plain mean.
    rm = float(
        np.median(rv)
    )

    gm = float(
        np.median(gv)
    )

    bm = float(
        np.median(bv)
    )

    avg = (
        rm + gm + bm
    ) / 3.0

    if min(
        rm,
        gm,
        bm,
    ) < 20:
        return (
            1.0,
            1.0,
            1.0,
        )

    r_gain = avg / rm
    g_gain = avg / gm
    b_gain = avg / bm

    r_gain = float(
        np.clip(
            r_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    g_gain = float(
        np.clip(
            g_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    b_gain = float(
        np.clip(
            b_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    return (
        r_gain,
        g_gain,
        b_gain,
    )


# ============================================================================
# LOCAL NEUTRAL COLOR ESTIMATION
# ============================================================================

def estimate_local_neutral_gains(
    analysis: ImageAnalysis,
    ring: np.ndarray,
    global_gains: Tuple[float, float, float],
) -> Tuple[
    Tuple[float, float, float],
    float,
]:

    image = analysis.image

    valid = (
        (ring > 0.05)
        &
        (analysis.luma > 0.12)
        &
        (analysis.luma < 0.90)
        &
        (analysis.saturation < 0.28)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 60:

        return (
            global_gains,
            0.0,
        )

    pixels = image[valid].astype(
        np.float32
    )

    b = pixels[:, 0]
    g = pixels[:, 1]
    r = pixels[:, 2]

    # Reject pixels whose channels are too different.
    mx = np.maximum.reduce(
        [r, g, b]
    )

    mn = np.minimum.reduce(
        [r, g, b]
    )

    ratio = (
        (mx - mn)
        / np.maximum(mx, 1.0)
    )

    good = ratio < 0.16

    if np.count_nonzero(good) < 40:

        return (
            global_gains,
            0.0,
        )

    r = r[good]
    g = g[good]
    b = b[good]

    # Use medians to avoid colored-object contamination.
    rm = float(
        np.median(r)
    )

    gm = float(
        np.median(g)
    )

    bm = float(
        np.median(b)
    )

    avg = (
        rm + gm + bm
    ) / 3.0

    if min(
        rm,
        gm,
        bm,
    ) < 15:

        return (
            global_gains,
            0.0,
        )

    local_r = avg / rm
    local_g = avg / gm
    local_b = avg / bm

    local_r = float(
        np.clip(
            local_r,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    local_g = float(
        np.clip(
            local_g,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    local_b = float(
        np.clip(
            local_b,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    # Confidence depends on sample count.
    confidence = float(
        np.clip(
            count / 1000.0,
            0.0,
            1.0,
        )
    )

    # Local correction is deliberately blended with global correction.
    blend = 0.30 * confidence

    result = (
        global_gains[0] * (1.0 - blend)
        + local_r * blend,

        global_gains[1] * (1.0 - blend)
        + local_g * blend,

        global_gains[2] * (1.0 - blend)
        + local_b * blend,
    )

    return (
        result,
        confidence,
    )


# ============================================================================
# APPLY NEUTRAL COLOR
# ============================================================================

def apply_neutral_color(
    roi: np.ndarray,
    gains: Tuple[float, float, float],
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    # Skin gets much weaker color correction.
    if profile.skin:
        amount *= 0.35

    if profile.reflective:
        amount *= 0.18

    amount *= profile.wb

    if amount <= 0:
        return roi.copy()

    b_gain = np.clip(
        1.0 + (gains[2] - 1.0) * amount,
        0.90,
        1.10,
    )

    g_gain = np.clip(
        1.0 + (gains[1] - 1.0) * amount,
        0.90,
        1.10,
    )

    r_gain = np.clip(
        1.0 + (gains[0] - 1.0) * amount,
        0.90,
        1.10,
    )

    luma_gain = (
        0.114 * b_gain
        + 0.587 * g_gain
        + 0.299 * r_gain
    )

    if luma_gain > 1e-6:
        b_gain /= luma_gain
        g_gain /= luma_gain
        r_gain /= luma_gain

    src = roi.astype(
        np.float32
    )

    result = src.copy()

    result[:, :, 0] *= b_gain
    result[:, :, 1] *= g_gain
    result[:, :, 2] *= r_gain

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# LAB NEUTRAL CAST PROTECTION
# ============================================================================

def apply_subtle_lab_neutralization(
    roi: np.ndarray,
    local_stats: Dict[str, float],
    ring_stats: Dict[str, float],
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    # This is NOT background color copying.
    #
    # It only removes a very small detected chromatic bias.
    da = (
        128.0
        - ring_stats.get(
            "lab_a",
            128.0,
        )
    )

    db = (
        128.0
        - ring_stats.get(
            "lab_b",
            128.0,
        )
    )

    # If surrounding area itself is almost neutral,
    # there is little evidence for a color cast.
    magnitude = math.sqrt(
        da * da
        + db * db
    )

    if magnitude < 1.5:
        return roi.copy()

    strength = (
        CONFIG.color_match_strength
        * profile.wb
        * amount
    )

    if profile.skin:
        strength *= 0.25

    if profile.reflective:
        strength *= 0.15

    da *= strength
    db *= strength

    da = float(
        np.clip(
            da,
            -CONFIG.max_a_change,
            CONFIG.max_a_change,
        )
    )

    db = float(
        np.clip(
            db,
            -CONFIG.max_b_change,
            CONFIG.max_b_change,
        )
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    lab[:, :, 1] += da
    lab[:, :, 2] += db

    lab[:, :, 1] = np.clip(
        lab[:, :, 1],
        0,
        255,
    )

    lab[:, :, 2] = np.clip(
        lab[:, :, 2],
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# EXPOSURE
# ============================================================================

def calculate_exposure(
    target: Dict[str, float],
    surrounding: Dict[str, float],
    profile: MaskProfile,
) -> float:

    target_p50 = target.get(
        "p50",
        target.get(
            "brightness",
            0.5,
        ),
    )

    surrounding_p50 = surrounding.get(
        "p50",
        surrounding.get(
            "brightness",
            target_p50,
        ),
    )

    difference = (
        surrounding_p50
        - target_p50
    ) * 255.0

    if abs(difference) <= CONFIG.exposure_dead_zone:
        return 0.0

    correction = (
        difference
        * CONFIG.exposure_match_strength
        * profile.exposure
    )

    if correction < 0:

        if profile.skin:
            correction *= (
                CONFIG.skin_negative_exposure_factor
            )
        else:
            correction *= 0.55

    else:

        if profile.skin:
            correction *= (
                CONFIG.skin_positive_exposure_factor
            )

    return float(
        np.clip(
            correction,
            -CONFIG.max_exposure_change * 0.45,
            CONFIG.max_exposure_change,
        )
    )


# ============================================================================
# SHADOW RECOVERY
# ============================================================================

def calculate_shadow_lift(
    stats: Dict[str, float],
    profile: MaskProfile,
) -> float:

    shadow_fraction = stats.get(
        "shadow_fraction",
        0.0,
    )

    p25 = stats.get(
        "p25",
        0.25,
    )

    danger = 0.0

    if p25 < 0.28:
        danger += (
            0.28 - p25
        ) / 0.28

    danger += (
        shadow_fraction
        * 1.25
    )

    danger = float(
        np.clip(
            danger,
            0.0,
            1.5,
        )
    )

    amount = (
        danger
        * CONFIG.shadow_strength
        * profile.recovery
        * CONFIG.max_shadow_lift
    )

    if profile.skin:
        amount *= 0.65

    return float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_shadow_lift,
        )
    )


# ============================================================================
# HIGHLIGHT RECOVERY
# ============================================================================

def calculate_highlight_reduction(
    stats: Dict[str, float],
    profile: MaskProfile,
) -> float:

    p90 = stats.get(
        "p90",
        0.8,
    )

    clipped = stats.get(
        "clipped_fraction",
        0.0,
    )

    danger = 0.0

    if p90 > 0.82:
        danger += (
            p90 - 0.82
        ) / 0.18

    danger += clipped * 3.0

    danger = float(
        np.clip(
            danger,
            0.0,
            1.5,
        )
    )

    amount = (
        danger
        * CONFIG.highlight_strength
        * profile.recovery
        * CONFIG.max_highlight_reduction
    )

    # Skin highlights are protected but not ignored.
    if profile.skin:
        amount *= 0.65

    return float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_highlight_reduction,
        )
    )


# ============================================================================
# TONAL ADJUSTMENT
# ============================================================================

def apply_tonal_adjustment(
    roi: np.ndarray,
    exposure: float,
    shadow_lift: float,
    highlight_reduction: float,
    contrast_amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0] * (
        255.0 / 255.0
    )

    # ------------------------------------------------------------------------
    # Exposure.
    # ------------------------------------------------------------------------

    if abs(exposure) > 0.001:

        L += (
            exposure
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Shadow lift.
    # ------------------------------------------------------------------------

    if shadow_lift > 0:

        normalized = L / 255.0

        shadow_weight = np.clip(
            (0.42 - normalized)
            / 0.42,
            0.0,
            1.0,
        )

        shadow_weight = (
            shadow_weight
            ** 1.35
        )

        L += (
            shadow_weight
            * shadow_lift
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Highlight compression.
    # ------------------------------------------------------------------------

    if highlight_reduction > 0:

        normalized = L / 255.0

        highlight_weight = np.clip(
            (normalized - 0.68)
            / 0.32,
            0.0,
            1.0,
        )

        highlight_weight = (
            highlight_weight
            ** 1.20
        )

        L -= (
            highlight_weight
            * highlight_reduction
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Adaptive contrast.
    # ------------------------------------------------------------------------

    if contrast_amount > 0:

        normalized = L / 255.0

        # Very gentle S curve.
        curve = (
            normalized
            + contrast_amount
            * (
                normalized
                - normalized * normalized
            )
        )

        L = (
            normalized * 255.0
            * 0.78
            +
            curve * 255.0
            * 0.22
        )

    L = np.clip(
        L,
        0,
        255,
    )

    lab[:, :, 0] = L

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# LOCAL CONTRAST
# ============================================================================

def apply_local_clarity(
    roi: np.ndarray,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.clarity_strength
        * profile.clarity
    )

    if amount <= 0:
        return roi.copy()

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0]

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        CONFIG.clarity_sigma,
    )

    detail = L - blur

    # Prevent strong clarity on extreme shadows/highlights.
    normalized = L / 255.0

    protection = (
        1.0
        -
        0.65
        *
        np.maximum(
            np.clip(
                0.12 - normalized,
                0,
                0.12,
            ) / 0.12,
            np.clip(
                normalized - 0.92,
                0,
                0.08,
            ) / 0.08,
        )
    )

    L += (
        detail
        * amount
        * protection
    )

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# VIBRANCE
# ============================================================================

def apply_vibrance(
    roi: np.ndarray,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.saturation_strength
        * profile.saturation
    )

    if amount <= 0:
        return roi.copy()

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)

    saturation = (
        hsv[:, :, 1]
        / 255.0
    )

    # Less saturated pixels get more boost.
    protection = (
        1.0
        - saturation
    )

    # Protect already highly saturated colors.
    boost = (
        1.0
        + amount
        * protection
    )

    if profile.skin:

        # Skin receives only a tiny saturation adjustment.
        boost = (
            1.0
            + (
                boost - 1.0
            ) * 0.35
        )

    hsv[:, :, 1] *= boost

    # Hard saturation protection.
    if profile.saturation_limit < 1.0:

        max_sat = (
            255.0
            * (
                0.75
                + 0.25
                * profile.saturation_limit
            )
        )

        hsv[:, :, 1] = np.minimum(
            hsv[:, :, 1],
            max_sat,
        )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255,
    )

    return cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )


# ============================================================================
# LUMINANCE SHARPENING
# ============================================================================

def apply_luminance_sharpening(
    roi: np.ndarray,
    sharpness: float,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.sharpening_strength
        * profile.sharpening
    )

    amount = float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_sharpen,
        )
    )

    if amount <= 0:
        return roi.copy()

    # Already sharp images receive less sharpening.
    if sharpness > 4500:
        amount *= 0.20

    elif sharpness > 2500:
        amount *= 0.40

    elif sharpness > 1200:
        amount *= 0.70

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0]

    if sharpness < 600:

        sigma = 0.85

    elif sharpness < 1400:

        sigma = 1.00

    else:

        sigma = 1.15

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        sigma,
    )

    detail = L - blur

    # Don't sharpen extreme highlights.
    normalized = L / 255.0

    protection = 1.0 - 0.55 * np.clip(
        (normalized - 0.88)
        / 0.12,
        0.0,
        1.0,
    )

    if profile.skin:
        amount *= 0.55

    L += (
        detail
        * amount
        * protection
    )

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def integrate_mask_with_surrounding(
    roi: np.ndarray,
    working: np.ndarray,
    target_stats: Dict[str, float],
    surrounding_stats: Dict[str, float],
    strength: float,
) -> np.ndarray:
    """Blend a local mask back toward the surrounding scene so it feels integrated."""

    ring_mean = float(surrounding_stats.get("brightness", 0.5))
    roi_mean = float(target_stats.get("brightness", ring_mean))
    working_gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY).astype(np.float32)
    working_mean = float(np.mean(working_gray) / 255.0)

    target_mean = np.clip(
        0.55 * ring_mean + 0.45 * roi_mean,
        0.10,
        0.90,
    )

    delta = target_mean - working_mean
    if abs(delta) < 0.02:
        return working

    amount = float(np.clip(0.25 + 0.45 * strength, 0.18, 0.70))

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L = np.clip(L + delta * 255.0 * amount, 0.0, 255.0)
    lab[:, :, 0] = L

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================================================
# MASK-SPECIFIC ENHANCEMENT
# ============================================================================

def enhance_region(
    image: np.ndarray,
    analysis: ImageAnalysis,
    mask: np.ndarray,
    surrounding: np.ndarray,
    target_stats: Dict[str, float],
    surrounding_stats: Dict[str, float],
    name: str,
    global_strength: float,
    global_gains: Tuple[float, float, float],
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:

    profile = MASK_PROFILES.get(
        name.lower(),
        DEFAULT_PROFILE,
    )

    pixels = target_stats["pixels"]

    if pixels < CONFIG.minimum_mask_pixels:

        return image.copy(), {
            "name": name,
            "pixels": pixels,
            "skipped": True,
            "reason": "too_few_pixels",
            "analysis": target_stats,
        }

    # ------------------------------------------------------------------------
    # Effective strength.
    # ------------------------------------------------------------------------

    strength = (
        profile.strength
        * global_strength
    )

    if pixels < 1500:
        strength *= 0.82

    elif pixels < 5000:
        strength *= 0.92

    strength = float(
        np.clip(
            strength,
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------------------------
    # Bounding box.
    # ------------------------------------------------------------------------

    bbox = mask_bbox(
        mask,
        padding=CONFIG.ring_outer_radius + 4,
    )

    if bbox is None:

        return image.copy(), {
            "name": name,
            "pixels": pixels,
            "skipped": True,
            "reason": "empty_mask",
            "analysis": target_stats,
        }

    x1, y1, x2, y2 = bbox

    roi = image[
        y1:y2,
        x1:x2,
    ]

    mask_roi = mask[
        y1:y2,
        x1:x2,
    ]

    # ------------------------------------------------------------------------
    # Local analysis.
    # ------------------------------------------------------------------------

    roi_analysis = analysis.crop(
        y1,
        y2,
        x1,
        x2,
    )

    target_roi_stats = analyze_region(
        roi_analysis,
        mask_roi,
    )

    ring_roi = surrounding[
        y1:y2,
        x1:x2,
    ]

    ring_stats = analyze_region(
        roi_analysis,
        ring_roi,
    )

    # ------------------------------------------------------------------------
    # Local neutral illumination.
    # ------------------------------------------------------------------------

    local_gains, wb_confidence = (
        estimate_local_neutral_gains(
            roi_analysis,
            ring_roi,
            global_gains,
        )
    )

    working = roi.copy()

    # ------------------------------------------------------------------------
    # White balance.
    # ------------------------------------------------------------------------

    working = apply_neutral_color(
        working,
        local_gains,
        wb_confidence,
        profile,
    )

    # ------------------------------------------------------------------------
    # Very subtle LAB cast correction.
    # ------------------------------------------------------------------------

    working = (
        apply_subtle_lab_neutralization(
            working,
            target_roi_stats,
            ring_stats,
            wb_confidence,
            profile,
        )
    )

    # ------------------------------------------------------------------------
    # Exposure.
    # ------------------------------------------------------------------------

    exposure = calculate_exposure(
        target_roi_stats,
        ring_stats,
        profile,
    )

    # ------------------------------------------------------------------------
    # Shadow / highlight recovery.
    # ------------------------------------------------------------------------

    shadow_lift = calculate_shadow_lift(
        target_roi_stats,
        profile,
    )

    highlight_reduction = (
        calculate_highlight_reduction(
            target_roi_stats,
            profile,
        )
    )

    # ------------------------------------------------------------------------
    # Adaptive contrast.
    # ------------------------------------------------------------------------

    target_contrast = (
        target_roi_stats["contrast"]
        * 255.0
    )

    surrounding_contrast = (
        ring_stats["contrast"]
        * 255.0
    )

    contrast_difference = (
        surrounding_contrast
        - target_contrast
    )

    contrast_amount = 0.0

    if abs(contrast_difference) > 5:

        contrast_amount = (
            contrast_difference
            / 255.0
            * CONFIG.contrast_strength
            * profile.contrast
        )

        contrast_amount = float(
            np.clip(
                contrast_amount,
                -0.10,
                0.14,
            )
        )

    # ------------------------------------------------------------------------
    # Tone.
    # ------------------------------------------------------------------------

    working = apply_tonal_adjustment(
        working,
        exposure,
        shadow_lift,
        highlight_reduction,
        contrast_amount,
        profile,
    )

    # ------------------------------------------------------------------------
    # Clarity.
    # ------------------------------------------------------------------------

    working = apply_local_clarity(
        working,
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Vibrance.
    # ------------------------------------------------------------------------

    working = apply_vibrance(
        working,
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Sharpening.
    # ------------------------------------------------------------------------

    working = apply_luminance_sharpening(
        working,
        target_roi_stats["sharpness"],
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Integrate the mask back into the surrounding scene so it does not feel
    # isolated, too dark, or too bright. This is the final step that keeps the
    # local correction natural and image-consistent.
    # ------------------------------------------------------------------------

    working = integrate_mask_with_surrounding(
        roi,
        working,
        target_roi_stats,
        ring_stats,
        strength,
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # No blending happens here.
    #
    # The compositor is the ONLY place that blends the mask.
    # ------------------------------------------------------------------------

    enhanced_full = image.copy()

    # Keep processed pixels inside the semantic mask. The padded ROI is only
    # context for local filters; writing it back in full creates rectangular
    # seams at the ROI boundary.
    mask_coverage = mask_roi > 0.01
    enhanced_roi = np.where(
        mask_coverage[:, :, None],
        working,
        roi,
    )

    enhanced_full[
        y1:y2,
        x1:x2,
    ] = enhanced_roi

    report = {
        "name": name,
        "pixels": pixels,
        "strength": strength,
        "skipped": False,

        "bbox": [
            x1,
            y1,
            x2,
            y2,
        ],

        "analysis": target_stats,
        "surrounding": surrounding_stats,

        "corrections": {
            "exposure": float(exposure),
            "shadow_lift": float(shadow_lift),
            "highlight_reduction": float(
                highlight_reduction
            ),
            "contrast": float(
                contrast_amount
            ),
            "wb_confidence": float(
                wb_confidence
            ),
            "local_gains": [
                float(local_gains[0]),
                float(local_gains[1]),
                float(local_gains[2]),
            ],
        },

        "profile": asdict(
            profile
        ),
    }

    return (
        enhanced_full,
        report,
    )


# ============================================================================
# SINGLE COMPOSITOR
# ============================================================================

def composite_mask(
    base: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    ownership: np.ndarray,
    strength: float,
    feather_radius: float,
) -> np.ndarray:

    alpha = np.clip(
        mask.astype(np.float32),
        0.0,
        1.0,
    )

    # Remove weak rectangular/background coverage from imperfect mask PNGs
    # before feathering. Feathering a low-alpha rectangle makes the artifact
    # visible as a translucent box around the intended subject.
    alpha[alpha < 0.05] = 0.0

    # Strength is applied exactly once.
    alpha *= float(
        np.clip(
            strength,
            0.0,
            1.0,
        )
    )

    # Feather exactly once.
    alpha = feather_mask(
        alpha,
        feather_radius,
    )

    # Prevent weaker overlapping masks from overwriting
    # stronger semantic regions.
    available = np.clip(
        1.0 - ownership,
        0.0,
        1.0,
    )

    alpha *= available

    alpha[
        alpha < 0.003
    ] = 0.0

    alpha3 = alpha[:, :, None]

    base_f = base.astype(
        np.float32,
        copy=False,
    )

    enhanced_f = enhanced.astype(
        np.float32,
        copy=False,
    )

    result = (
        base_f
        * (1.0 - alpha3)
        +
        enhanced_f
        * alpha3
    )

    ownership[:] = np.maximum(
        ownership,
        alpha,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# SORTING
# ============================================================================

def mask_sort_key(
    name: str,
) -> int:

    name = name.lower()

    try:
        return MASK_PRIORITY.index(
            name
        )

    except ValueError:
        return (
            len(MASK_PRIORITY)
            + 100
        )


# ============================================================================
# PERSON MASK SUPPORT
# ============================================================================

def discover_person_masks(
    person_masks_dir: Optional[Path],
    width: int,
    height: int,
) -> Dict[
    str,
    Dict[str, np.ndarray],
]:

    if person_masks_dir is None:
        return {}

    if not person_masks_dir.exists():
        return {}

    result: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    for person_root in sorted(
        person_masks_dir.glob(
            "person_*"
        )
    ):

        if not person_root.is_dir():
            continue

        person_id = (
            person_root.name
        )

        soft_dir = (
            person_root / "soft"
        )

        binary_dir = (
            person_root / "binary"
        )

        source_dir = (
            soft_dir
            if soft_dir.exists()
            else binary_dir
        )

        if not source_dir.exists():
            continue

        person_masks: Dict[
            str,
            np.ndarray,
        ] = {}

        for path in source_dir.glob(
            "*.png"
        ):

            name = path.stem.lower()

            mask = load_mask(
                path,
                width,
                height,
            )

            if mask is None:
                continue

            mask = clean_mask(
                mask
            )

            if (
                np.count_nonzero(
                    mask > 0.05
                )
                <
                CONFIG.minimum_mask_pixels
            ):
                continue

            person_masks[name] = mask

        if person_masks:
            result[
                person_id
            ] = person_masks

    return result


# ============================================================================
# PROCESS ONE SET OF MASKS
# ============================================================================

def process_mask_set(
    image: np.ndarray,
    analysis: ImageAnalysis,
    loaded_masks: Dict[str, np.ndarray],
    final_image: np.ndarray,
    ownership: np.ndarray,
    global_strength: float,
    feather_radius: float,
    global_gains: Tuple[float, float, float],
    reports: List[Dict[str, Any]],
    save_mask_results: Optional[Path],
    progress: Optional[
        Callable[[int, str], None]
    ] = None,
    progress_start: int = 25,
    progress_end: int = 90,
) -> np.ndarray:

    if not loaded_masks:
        return final_image

    # Union of all masks.
    all_masks = np.zeros_like(
        next(iter(loaded_masks.values())),
        dtype=np.float32,
    )

    for mask in loaded_masks.values():
        all_masks = np.maximum(
            all_masks,
            mask,
            out=all_masks,
        )

    ordered_names = sorted(
        loaded_masks.keys(),
        key=mask_sort_key,
    )

    total = len(
        ordered_names
    )

    for index, name in enumerate(
        ordered_names,
        start=1,
    ):

        if progress is not None:

            percent = (
                progress_start
                +
                int(
                    index
                    * (
                        progress_end
                        - progress_start
                    )
                    /
                    max(
                        total,
                        1,
                    )
                )
            )

            progress(
                percent,
                f"enhance-mask:{name}",
            )

        mask = loaded_masks[
            name
        ]

        pixels = int(
            np.count_nonzero(
                mask > 0.05
            )
        )

        logger.info(
            "[%d/%d] %s pixels=%d",
            index,
            total,
            name,
            pixels,
        )

        # --------------------------------------------------------------------
        # Context.
        # --------------------------------------------------------------------

        surrounding_stats, ring = (
            analyze_surrounding(
                analysis,
                mask,
                all_masks=all_masks,
            )
        )

        # --------------------------------------------------------------------
        # Target.
        # --------------------------------------------------------------------

        target_stats = analyze_region(
            analysis,
            mask,
        )

        # --------------------------------------------------------------------
        # Enhance ORIGINAL image.
        # --------------------------------------------------------------------

        enhanced, report = (
            enhance_region(
                image,
                analysis,
                mask,
                ring,
                target_stats,
                surrounding_stats,
                name,
                global_strength,
                global_gains,
            )
        )

        reports.append(
            report
        )

        if report.get(
            "skipped",
            False,
        ):
            continue

        profile = MASK_PROFILES.get(
            name.lower(),
            DEFAULT_PROFILE,
        )

        effective_strength = float(
            np.clip(
                profile.strength
                * global_strength,
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------------
        # SINGLE COMPOSITE.
        # --------------------------------------------------------------------

        final_image = composite_mask(
            final_image,
            enhanced,
            mask,
            ownership,
            effective_strength,
            feather_radius,
        )

        # --------------------------------------------------------------------
        # Optional preview.
        # --------------------------------------------------------------------

        if save_mask_results is not None:

            preview = image.copy()

            preview_ownership = (
                np.zeros_like(
                    ownership
                )
            )

            preview = composite_mask(
                preview,
                enhanced,
                mask,
                preview_ownership,
                effective_strength,
                feather_radius,
            )

            preview_path = (
                save_mask_results
                /
                f"{name}_enhanced.png"
            )

            imwrite_unicode(
                preview_path,
                preview,
            )

    return final_image


# ============================================================================
# MAIN PROCESS
# ============================================================================

def process(
    image_path: Path,
    json_path: Path,
    output_path: Path,
    mask_dir: Optional[Path] = None,
    global_strength: float = 1.0,
    feather_radius: float = 2.0,
    save_mask_results: Optional[Path] = None,
    progress: Optional[
        Callable[[int, str], None]
    ] = None,
    person_masks_dir: Optional[Path] = None,
    mask_names: Optional[List[str]] = None,
) -> None:

    def report_progress(
        percent: int,
        stage: str,
    ) -> None:

        if progress is not None:
            progress(
                max(
                    0,
                    min(
                        100,
                        int(percent),
                    ),
                ),
                stage,
            )

    # ========================================================================
    # LOAD IMAGE
    # ========================================================================

    report_progress(
        5,
        "load-image",
    )

    logger.info(
        "Loading image: %s",
        image_path,
    )

    image = imread_unicode(
        image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape[:2]

    logger.info(
        "Image size: %dx%d",
        width,
        height,
    )

    # Immutable original.
    source = image.copy()

    # Final compositor.
    final_image = image.copy()

    # ========================================================================
    # JSON
    # ========================================================================

    report_progress(
        10,
        "load-json",
    )

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # ========================================================================
    # MASK DISCOVERY
    # ========================================================================

    report_progress(
        18,
        "discover-masks",
    )

    masks = extract_mask_entries(
        data,
        json_path,
        mask_dir,
    )

    # Explicit mask directory wins.
    if mask_dir is not None:

        directory_masks = (
            discover_masks_from_directory(
                mask_dir
            )
        )

        masks.update(
            directory_masks
        )

    # ========================================================================
    # VALIDATE
    # ========================================================================

    clean_masks: Dict[
        str,
        Path,
    ] = {}

    image_resolved = (
        image_path.resolve()
    )

    for name, path in masks.items():

        name = (
            str(name)
            .strip()
            .lower()
        )

        try:

            if (
                path.resolve()
                ==
                image_resolved
            ):
                continue

        except Exception:
            continue

        if path.suffix.lower() != ".png":
            continue

        if not path.exists():
            continue

        clean_masks[
            name
        ] = path

    masks = clean_masks

    if mask_names is not None:
        requested_names = {
            str(name).strip().lower()
            for name in mask_names
            if str(name).strip()
        }
        masks = {
            name: path
            for name, path in masks.items()
            if name in requested_names
        }

    if not masks:

        raise RuntimeError(
            "No valid PNG masks were found."
        )

    # ========================================================================
    # LOAD MASKS
    # ========================================================================

    report_progress(
        22,
        "load-masks",
    )

    loaded_masks: Dict[
        str,
        np.ndarray,
    ] = {}

    for name, path in sorted(
        masks.items(),
        key=lambda item: mask_sort_key(
            item[0]
        ),
    ):

        mask = load_mask(
            path,
            width,
            height,
        )

        if mask is None:
            continue

        mask = clean_mask(
            mask
        )

        pixel_count = int(
            np.count_nonzero(
                mask > 0.05
            )
        )

        if (
            pixel_count
            <
            CONFIG.minimum_mask_pixels
        ):
            continue

        loaded_masks[
            name
        ] = mask

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    report_progress(
        25,
        "analyze-image",
    )

    logger.info(
        "Building image analysis cache..."
    )

    analysis = ImageAnalysis(
        source
    )

    # ========================================================================
    # GLOBAL NEUTRAL ILLUMINANT
    # ========================================================================

    report_progress(
        27,
        "estimate-neutral-light",
    )

    global_gains = (
        estimate_global_neutral_gains(
            analysis
        )
    )

    logger.info(
        "Global neutral gains: "
        "R=%.4f G=%.4f B=%.4f",
        global_gains[0],
        global_gains[1],
        global_gains[2],
    )

    # ========================================================================
    # OWNERSHIP
    # ========================================================================

    ownership = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    reports: List[
        Dict[str, Any]
    ] = []

    if save_mask_results is not None:

        save_mask_results.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # GLOBAL MASK PROCESSING
    # ========================================================================

    if loaded_masks:

        final_image = process_mask_set(
            image=source,
            analysis=analysis,
            loaded_masks=loaded_masks,
            final_image=final_image,
            ownership=ownership,
            global_strength=float(
                np.clip(
                    global_strength,
                    0.0,
                    1.0,
                )
            ),
            feather_radius=max(
                0.0,
                float(
                    feather_radius
                ),
            ),
            global_gains=global_gains,
            reports=reports,
            save_mask_results=save_mask_results,
            progress=progress,
            progress_start=28,
            progress_end=88,
        )

    # ========================================================================
    # PERSON MASK SUPPORT
    # ========================================================================

    person_masks = (
        discover_person_masks(
            person_masks_dir,
            width,
            height,
        )
    )

    if person_masks:

        logger.info(
            "Found %d person mask groups.",
            len(person_masks),
        )

        person_root_reports = []

        for person_id, person_group in (
            person_masks.items()
        ):

            logger.info(
                "Processing %s",
                person_id,
            )

            before_count = len(
                reports
            )

            final_image = (
                process_mask_set(
                    image=source,
                    analysis=analysis,
                    loaded_masks=person_group,
                    final_image=final_image,
                    ownership=ownership,
                    global_strength=float(
                        np.clip(
                            global_strength,
                            0.0,
                            1.0,
                        )
                    ),
                    feather_radius=max(
                        0.0,
                        float(
                            feather_radius
                        ),
                    ),
                    global_gains=global_gains,
                    reports=reports,
                    save_mask_results=None,
                    progress=progress,
                    progress_start=40,
                    progress_end=88,
                )
            )

            after_count = len(
                reports
            )

            person_root_reports.append(
                {
                    "person_id": person_id,
                    "reports": reports[
                        before_count:
                        after_count
                    ],
                }
            )

        # Keep report information.
        reports.append(
            {
                "person_groups": person_root_reports
            }
        )

    # ========================================================================
    # FINAL GLOBAL SAFETY
    # ========================================================================

    report_progress(
        92,
        "finalize",
    )

    # Avoid any accidental NaN/Inf.
    final_image = np.nan_to_num(
        final_image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    final_image = np.clip(
        final_image,
        0,
        255,
    ).astype(np.uint8)

    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================

    report_progress(
        95,
        "save-output",
    )

    logger.info(
        "Saving output: %s",
        output_path,
    )

    if not imwrite_unicode(
        output_path,
        final_image,
        quality=95,
    ):

        raise RuntimeError(
            f"Could not save output: {output_path}"
        )

    # ========================================================================
    # REPORT
    # ========================================================================

    report_progress(
        98,
        "save-report",
    )

    report_path = (
        output_path.parent
        /
        f"{output_path.stem}_mask_report.json"
    )

    report_data = {
        "engine":
            "humanmaskextractor_adaptive_neutral_v6",

        "input":
            str(image_path),

        "mask_json":
            str(json_path),

        "mask_dir":
            str(mask_dir)
            if mask_dir
            else None,

        "person_masks_dir":
            str(person_masks_dir)
            if person_masks_dir
            else None,

        "output":
            str(output_path),

        "image_width":
            width,

        "image_height":
            height,

        "global_strength":
            float(global_strength),

        "feather_radius":
            float(feather_radius),

        "global_neutral_gains": [
            float(global_gains[0]),
            float(global_gains[1]),
            float(global_gains[2]),
        ],

        "masks_processed":
            sorted(
                loaded_masks.keys(),
                key=mask_sort_key,
            ),

        "reports":
            reports,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Report saved: %s",
        report_path,
    )

    report_progress(
        100,
        "completed",
    )

    logger.info(
        "DONE: %s",
        output_path,
    )


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive semantic-mask photo "
            "enhancement with neutral illumination "
            "correction."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Original input image",
    )

    parser.add_argument(
        "masks_json",
        type=Path,
        help="Mask JSON",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Enhanced output image",
    )

    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing PNG masks."
        ),
    )

    parser.add_argument(
        "--person-masks-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing "
            "person_001/soft, person_002/soft, etc."
        ),
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Global enhancement strength "
            "(0.0-1.0). Default: 1.0"
        ),
    )

    parser.add_argument(
        "--feather",
        type=float,
        default=2.0,
        help=(
            "Mask feather radius in pixels. "
            "Default: 2.0"
        ),
    )

    parser.add_argument(
        "--save-mask-results",
        type=Path,
        default=None,
        help=(
            "Directory for individual mask "
            "previews."
        ),
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

def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    try:

        strength = float(
            np.clip(
                args.strength,
                0.0,
                1.0,
            )
        )

        feather = max(
            0.0,
            float(
                args.feather
            ),
        )

        process(
            image_path=args.input,
            json_path=args.masks_json,
            output_path=args.output,
            mask_dir=args.mask_dir,
            global_strength=strength,
            feather_radius=feather,
            save_mask_results=(
                args.save_mask_results
            ),
            person_masks_dir=(
                args.person_masks_dir
            ),
        )

        return 0

    except KeyboardInterrupt:

        logger.error(
            "Interrupted by user."
        )

        return 130

    except Exception as exc:

        logger.exception(
            "ERROR: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())