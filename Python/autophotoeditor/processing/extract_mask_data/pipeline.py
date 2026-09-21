from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
import logging
import math

import cv2
import numpy as np

from .shared import *
from .masks import *
from .analysis import *
from .profiles import *


logger = logging.getLogger(__name__)


def choose_device(device: str = "auto") -> str:
    """Normalize device selection for mask extraction."""
    value = (device or "auto").strip().lower()
    if value in {"cpu", "cuda"}:
        if value == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_pipeline(
    *,
    input_path: str | Path,
    output_json: str | Path,
    person_model: str | None = None,
    human_model: str | None = None,
    face_model: str | None = None,
    device: str = "auto",
    person_imgsz: int = 1280,
    person_conf: float = 0.25,
    person_iou: float = 0.45,
    face_threshold: float = 0.5,
    face_padding: float = 0.10,
    visual_path: str | Path | None = None,
    visual_dir: str | Path | None = None,
    binary_dir: str | Path | None = None,
    soft_dir: str | Path | None = None,
    person_masks_dir: str | Path | None = None,
    mask_opacity: float = 0.6,
    jpeg_quality: int = 95,
    pretty: bool = True,
    no_rle: bool = False,
    save_person_previews: bool = False,
    feather_sigma: float = 1.5,
    **_: Any,
) -> tuple[Path, Path]:
    """Compatibility wrapper used by the API and WPF flow.

    The mask-extraction component is intentionally lightweight here: it creates a
    valid mask bundle and JSON for the downstream masked-adaptive enhancement
    pipeline without reintroducing the unused legacy modules.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")

    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_root = Path(binary_dir).expanduser().resolve() if binary_dir is not None else output_path.parent / "binary_masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {source}")

    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    mask_path = mask_root / f"{source.stem}_mask.png"
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Could not write default mask: {mask_path}")

    mask_names = {
        "subject": mask,
        "foreground": mask,
    }

    recommendations = AutoMaskEnhancer(
        image,
        hard_masks=mask_names,
        soft_masks={},
        global_strength=1.0,
    ).analyze()

    payload = {
        "masks": {
            name: {
                "binary_file": str(
                    mask_path.relative_to(
                        output_path.parent
                    )
                ),
                "enhancement": recommendations.get(
                    name,
                    {},
                ),
            }
            for name in mask_names
        }
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if pretty else None)
        if pretty:
            handle.write("\n")

    return output_path, mask_root


# ============================================================
# CONSTANTS
# ============================================================

OUTPUT_VERSION = "2.0"

EPSILON = 1e-8

# Global tone safety.
GLOBAL_EXPOSURE_LIMIT = 0.75
GLOBAL_CONTRAST_LIMIT = 15.0
GLOBAL_HIGHLIGHT_LIMIT = 25.0
GLOBAL_SHADOW_LIMIT = 25.0
GLOBAL_WHITE_LIMIT = 15.0
GLOBAL_BLACK_LIMIT = 10.0

# Small corrections below this are considered visually
# insignificant and are removed.
GLOBAL_ZERO_THRESHOLD = 0.01


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(value):
        return default

    return value


def _finite(
    value: Any,
) -> float:

    return _safe_float(
        value,
        0.0,
    )


def _safe_clip(
    value: Any,
    low: float,
    high: float,
) -> float:

    value = _safe_float(value)

    low = _safe_float(low)
    high = _safe_float(high)

    if low > high:
        low, high = high, low

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


# ============================================================
# GLOBAL ENHANCEMENT
# ============================================================

def calculate_global_enhancement(
    data: ImageColorData,
    stats: GlobalSceneStats,
) -> EnhancementValues:

    """
    Calculate conservative global tone corrections.

    This function intentionally does NOT attempt semantic
    corrections. Those belong to the mask-specific pipeline.

    The global pass should establish a sensible overall
    exposure/tonal foundation without fighting local edits.
    """

    median = max(
        _safe_float(stats.p50),
        0.001,
    )

    p05 = _safe_float(stats.p05)
    p10 = _safe_float(stats.p10)
    p95 = _safe_float(stats.p95)
    p99 = _safe_float(stats.p99)

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    exposure = math.log2(
        0.18 / median
    )

    # High-key protection.
    #
    # If the image already contains a lot of bright content,
    # do not aggressively move its global median toward 18%.
    if p95 > 0.92:

        exposure *= 0.50

    elif p95 > 0.85:

        exposure *= 0.75

    # Protect strongly low-key photographs.
    #
    # A low median does not necessarily mean the image is
    # incorrectly exposed.
    if (
        median < 0.08
        and
        p95 < 0.45
    ):

        exposure *= 0.65

    exposure = _safe_clip(
        exposure,
        -GLOBAL_EXPOSURE_LIMIT,
        GLOBAL_EXPOSURE_LIMIT,
    )

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    highlights = 0.0

    if p95 > 0.94:

        highlights = -_safe_clip(
            (
                p95
                -
                0.94
            )
            * 160.0,
            0.0,
            GLOBAL_HIGHLIGHT_LIMIT,
        )

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    shadows = 0.0

    if p10 < 0.035:

        shadows = _safe_clip(
            (
                0.035
                -
                p10
            )
            * 280.0,
            0.0,
            GLOBAL_SHADOW_LIMIT,
        )

    # Avoid lifting shadows excessively when the entire image
    # is intentionally dark.
    if (
        median < 0.10
        and
        p95 < 0.50
    ):

        shadows *= 0.65

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    dynamic_range = max(
        p95 - p05,
        0.0,
    )

    contrast = (
        dynamic_range
        -
        0.65
    ) * 45.0

    contrast = _safe_clip(
        contrast,
        -GLOBAL_CONTRAST_LIMIT,
        GLOBAL_CONTRAST_LIMIT,
    )

    # Very compressed photographs may need more contrast.
    if dynamic_range < 0.35:

        contrast = min(
            contrast + 4.0,
            GLOBAL_CONTRAST_LIMIT,
        )

    # Already very contrasty photographs should not receive
    # additional global contrast.
    if dynamic_range > 0.85:

        contrast = min(
            contrast,
            3.0,
        )

    # --------------------------------------------------------
    # Whites
    # --------------------------------------------------------

    whites = 0.0

    if p99 > 0.985:

        whites = -_safe_clip(
            (
                p99
                -
                0.985
            )
            * 150.0,
            0.0,
            GLOBAL_WHITE_LIMIT,
        )

    # --------------------------------------------------------
    # Blacks
    # --------------------------------------------------------

    blacks = 0.0

    if (
        p05 > 0.025
        and
        p10 > 0.055
    ):

        # Only gently lower blacks when there is no meaningful
        # deep-shadow information.
        blacks = -_safe_clip(
            (
                p05
                -
                0.025
            )
            * 80.0,
            0.0,
            GLOBAL_BLACK_LIMIT,
        )

    elif p01 := _safe_float(
        getattr(
            stats,
            "p01",
            0.0,
        )
    ) < 0.008:

        blacks = _safe_clip(
            (
                0.008
                -
                p01
            )
            * 450.0,
            0.0,
            GLOBAL_BLACK_LIMIT,
        )

    # --------------------------------------------------------
    # Final sanitization
    # --------------------------------------------------------

    values = EnhancementValues(
        exposure=exposure,
        contrast=contrast,
        highlights=highlights,
        shadows=shadows,
        whites=whites,
        blacks=blacks,
    )

    values = _limit_global_values(
        values
    )

    values = _remove_tiny_global_values(
        values
    )

    return values


# ============================================================
# GLOBAL LIMITS
# ============================================================

def _limit_global_values(
    values: EnhancementValues,
) -> EnhancementValues:

    result = {}

    for field_info in fields(values):

        name = field_info.name

        value = _finite(
            getattr(
                values,
                name,
                0.0,
            )
        )

        if name == "exposure":

            value = _safe_clip(
                value,
                -GLOBAL_EXPOSURE_LIMIT,
                GLOBAL_EXPOSURE_LIMIT,
            )

        elif name == "contrast":

            value = _safe_clip(
                value,
                -GLOBAL_CONTRAST_LIMIT,
                GLOBAL_CONTRAST_LIMIT,
            )

        elif name == "highlights":

            value = _safe_clip(
                value,
                -GLOBAL_HIGHLIGHT_LIMIT,
                GLOBAL_HIGHLIGHT_LIMIT,
            )

        elif name == "shadows":

            value = _safe_clip(
                value,
                -GLOBAL_SHADOW_LIMIT,
                GLOBAL_SHADOW_LIMIT,
            )

        elif name == "whites":

            value = _safe_clip(
                value,
                -GLOBAL_WHITE_LIMIT,
                GLOBAL_WHITE_LIMIT,
            )

        elif name == "blacks":

            value = _safe_clip(
                value,
                -GLOBAL_BLACK_LIMIT,
                GLOBAL_BLACK_LIMIT,
            )

        else:

            limits = LIMITS.get(
                name,
                (-1.0, 1.0),
            )

            value = _safe_clip(
                value,
                limits[0],
                limits[1],
            )

        result[name] = value

    return EnhancementValues(
        **result
    )


def _remove_tiny_global_values(
    values: EnhancementValues,
) -> EnhancementValues:

    result = {}

    for field_info in fields(values):

        name = field_info.name

        value = _finite(
            getattr(
                values,
                name,
                0.0,
            )
        )

        if abs(value) < GLOBAL_ZERO_THRESHOLD:

            value = 0.0

        result[name] = value

    return EnhancementValues(
        **result
    )


# ============================================================
# MASK