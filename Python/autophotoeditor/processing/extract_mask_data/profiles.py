from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Iterable
import math

import numpy as np

from .shared import *
from .masks import *
from .analysis import *


# ============================================================
# ENHANCEMENT VALUES
# ============================================================

@dataclass
class EnhancementValues:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0

    temperature: float = 0.0
    tint: float = 0.0

    saturation: float = 0.0
    vibrance: float = 0.0

    clarity: float = 0.0
    dehaze: float = 0.0


@dataclass
class EnhancementRecommendation:
    values: EnhancementValues

    confidence: float = 0.0
    mask_quality: float = 0.0
    context_match: float = 0.0

    wb_confidence: float = 0.0
    tone_confidence: float = 0.0
    color_confidence: float = 0.0

    should_apply: bool = True


# ============================================================
# SAFETY CONSTANTS
# ============================================================

EPSILON = 1e-8

DEFAULT_MASK_STRENGTH = 0.20

MIN_PIXELS = {
    "eyes": 24,
    "lips": 24,
    "mouth": 24,
    "brows": 24,
    "face": 48,
    "skin": 64,
    "skin_face": 64,
    "skin_neck": 64,
    "hair": 64,
    "glasses": 24,
    "jewelry": 24,
    "clothing": 100,
    "top": 100,
    "dress": 100,
    "skirt": 100,
    "pants": 100,
    "torso": 100,
    "person": 100,
    "subject": 100,
    "foreground": 100,
}

MASK_STRENGTH = {
    "eyes": 0.80,
    "lips": 0.65,
    "mouth": 0.65,
    "brows": 0.70,
    "face": 0.75,
    "skin": 0.65,
    "skin_face": 0.65,
    "skin_neck": 0.60,
    "hair": 0.65,
    "glasses": 0.55,
    "jewelry": 0.55,
    "clothing": 0.55,
    "top": 0.55,
    "dress": 0.55,
    "skirt": 0.55,
    "pants": 0.55,
    "torso": 0.55,
    "person": 0.45,
    "subject": 0.45,
    "foreground": 0.45,
}

NO_OP_THRESHOLDS = {
    "exposure": 0.015,
    "contrast": 0.015,
    "highlights": 0.015,
    "shadows": 0.015,
    "whites": 0.015,
    "blacks": 0.015,

    "temperature": 0.50,
    "tint": 0.50,

    "saturation": 0.015,
    "vibrance": 0.015,

    "clarity": 0.010,
    "dehaze": 0.010,
}

LIMITS = {
    "exposure": (-1.50, 1.50),
    "contrast": (-20.0, 20.0),
    "highlights": (-50.0, 0.0),
    "shadows": (0.0, 40.0),
    "whites": (-30.0, 20.0),
    "blacks": (-20.0, 10.0),
    "temperature": (-20.0, 20.0),
    "tint": (-15.0, 15.0),
    "saturation": (-20.0, 20.0),
    "vibrance": (-15.0, 15.0),
    "clarity": (0.0, 0.35),
    "dehaze": (0.0, 12.0),
}


# ============================================================
# MASK-SPECIFIC EDIT BUDGETS
#
# These are deliberately conservative.
# The goal is to prevent segmentation or analysis mistakes
# from producing visibly aggressive edits.
# ============================================================

MASK_EDIT_BUDGETS: Dict[str, Dict[str, float]] = {

    "face": {
        "exposure": 0.30,
        "contrast": 0.15,
        "highlights": 0.20,
        "shadows": 0.20,
        "whites": 0.15,
        "blacks": 0.15,
        "temperature": 8.0,
        "tint": 6.0,
        "saturation": 0.08,
        "vibrance": 0.06,
        "clarity": 0.04,
        "dehaze": 0.02,
    },

    "skin": {
        "exposure": 0.25,
        "contrast": 0.10,
        "highlights": 0.18,
        "shadows": 0.18,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 6.0,
        "tint": 5.0,
        "saturation": 0.06,
        "vibrance": 0.04,
        "clarity": 0.025,
        "dehaze": 0.015,
    },

    "skin_face": {
        "exposure": 0.25,
        "contrast": 0.10,
        "highlights": 0.18,
        "shadows": 0.18,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 6.0,
        "tint": 5.0,
        "saturation": 0.06,
        "vibrance": 0.04,
        "clarity": 0.025,
        "dehaze": 0.015,
    },

    "skin_neck": {
        "exposure": 0.22,
        "contrast": 0.08,
        "highlights": 0.16,
        "shadows": 0.16,
        "whites": 0.10,
        "blacks": 0.10,
        "temperature": 5.0,
        "tint": 4.0,
        "saturation": 0.05,
        "vibrance": 0.035,
        "clarity": 0.02,
        "dehaze": 0.01,
    },

    "eyes": {
        "exposure": 0.12,
        "contrast": 0.12,
        "highlights": 0.08,
        "shadows": 0.08,
        "whites": 0.08,
        "blacks": 0.08,
        "temperature": 3.0,
        "tint": 2.0,
        "saturation": 0.04,
        "vibrance": 0.03,
        "clarity": 0.05,
        "dehaze": 0.015,
    },

    "lips": {
        "exposure": 0.10,
        "contrast": 0.08,
        "highlights": 0.08,
        "shadows": 0.08,
        "whites": 0.06,
        "blacks": 0.06,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.04,
        "vibrance": 0.025,
        "clarity": 0.025,
        "dehaze": 0.01,
    },

    "mouth": {
        "exposure": 0.10,
        "contrast": 0.08,
        "highlights": 0.08,
        "shadows": 0.08,
        "whites": 0.06,
        "blacks": 0.06,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.04,
        "vibrance": 0.025,
        "clarity": 0.025,
        "dehaze": 0.01,
    },

    "hair": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.12,
        "shadows": 0.15,
        "whites": 0.10,
        "blacks": 0.10,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.04,
        "vibrance": 0.03,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "clothing": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "top": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "dress": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "skirt": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "pants": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "torso": {
        "exposure": 0.15,
        "contrast": 0.15,
        "highlights": 0.15,
        "shadows": 0.15,
        "whites": 0.12,
        "blacks": 0.12,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.05,
        "vibrance": 0.04,
        "clarity": 0.06,
        "dehaze": 0.025,
    },

    "jewelry": {
        "exposure": 0.10,
        "contrast": 0.10,
        "highlights": 0.08,
        "shadows": 0.08,
        "whites": 0.08,
        "blacks": 0.08,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.03,
        "vibrance": 0.02,
        "clarity": 0.04,
        "dehaze": 0.02,
    },

    "glasses": {
        "exposure": 0.08,
        "contrast": 0.08,
        "highlights": 0.06,
        "shadows": 0.06,
        "whites": 0.06,
        "blacks": 0.06,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.02,
        "vibrance": 0.02,
        "clarity": 0.03,
        "dehaze": 0.015,
    },

    "brows": {
        "exposure": 0.08,
        "contrast": 0.08,
        "highlights": 0.06,
        "shadows": 0.06,
        "whites": 0.06,
        "blacks": 0.06,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.02,
        "vibrance": 0.02,
        "clarity": 0.03,
        "dehaze": 0.015,
    },

    "person": {
        "exposure": 0.12,
        "contrast": 0.10,
        "highlights": 0.10,
        "shadows": 0.10,
        "whites": 0.08,
        "blacks": 0.08,
        "temperature": 0.0,
        "tint": 0.0,
        "saturation": 0.03,
        "vibrance": 0.025,
        "clarity": 0.025,
        "dehaze": 0.01,
    },
}


# ============================================================
# MASK PRIORITY
#
# Higher number = more specific.
# Used later by the enhancement stage when resolving overlaps.
# ============================================================

MASK_PRIORITY = {
    "eyes": 100,
    "lips": 95,
    "mouth": 95,
    "brows": 90,

    "face": 85,

    "skin_face": 80,
    "skin_neck": 78,
    "skin": 75,

    "glasses": 70,
    "jewelry": 68,

    "hair": 60,

    "dress": 50,
    "skirt": 50,
    "pants": 50,
    "top": 50,
    "torso": 50,
    "clothing": 48,

    "person": 30,

    "background": 0,
    "non_person": 0,
}


# ============================================================
# WHITE BALANCE ELIGIBILITY
# ============================================================

WB_ALLOWED_MASKS = {
    "face",
    "skin",
    "skin_face",
    "skin_neck",
}

WB_WEAK_MASKS = {
    "eyes",
    "jewelry",
}

WB_DISABLED_MASKS = {
    "lips",
    "mouth",
    "hair",
    "clothing",
    "top",
    "dress",
    "skirt",
    "pants",
    "torso",
    "glasses",
    "brows",
    "person",
}


# ============================================================
# NUMERICAL SAFETY
# ============================================================

def _finite_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)
    except Exception:
        return default

    if not math.isfinite(value):
        return default

    return value


def _safe_clip(
    value: Any,
    low: float,
    high: float,
) -> float:

    value = _finite_float(value)

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


# ============================================================
# LIMIT VALUES
# ============================================================

def limit_values(
    values: EnhancementValues,
) -> EnhancementValues:

    result: Dict[str, float] = {}

    for field_name in values.__dataclass_fields__:

        value = getattr(
            values,
            field_name,
            0.0,
        )

        value = _finite_float(
            value,
            0.0,
        )

        limits = LIMITS.get(
            field_name
        )

        if limits is None:
            result[field_name] = value
            continue

        low, high = limits

        result[field_name] = _safe_clip(
            value,
            low,
            high,
        )

    return EnhancementValues(
        **result
    )


# ============================================================
# APPLY STRENGTH
# ============================================================

def apply_strength(
    values: EnhancementValues,
    strength: float,
) -> EnhancementValues:

    strength = _finite_float(
        strength,
        0.0,
    )

    strength = float(
        np.clip(
            strength,
            0.0,
            2.0,
        )
    )

    result: Dict[str, float] = {}

    for field_name in values.__dataclass_fields__:

        value = _finite_float(
            getattr(
                values,
                field_name,
                0.0,
            )
        )

        result[field_name] = (
            value * strength
        )

    return EnhancementValues(
        **result
    )


# ============================================================
# MASK-SPECIFIC BUDGET
# ============================================================

def apply_edit_budget(
    values: EnhancementValues,
    name: str,
    confidence: float,
) -> EnhancementValues:

    budget = MASK_EDIT_BUDGETS.get(
        name
    )

    if budget is None:
        budget = MASK_EDIT_BUDGETS.get(
            "person",
            {},
        )

    confidence = float(
        np.clip(
            _finite_float(
                confidence,
                0.0,
            ),
            0.0,
            1.0,
        )
    )

    result: Dict[str, float] = {}

    for field_name in values.__dataclass_fields__:

        value = _finite_float(
            getattr(
                values,
                field_name,
                0.0,
            )
        )

        maximum = budget.get(
            field_name
        )

        if maximum is None:
            result[field_name] = value
            continue

        # Confidence affects the permitted correction.
        #
        # Never completely disable a valid high-quality
        # recommendation solely because confidence is imperfect.
        effective_budget = (
            maximum
            * (
                0.35
                +
                0.65 * confidence
            )
        )

        result[field_name] = float(
            np.clip(
                value,
                -effective_budget,
                effective_budget,
            )
        )

    return EnhancementValues(
        **result
    )


# ============================================================
# NO-OP FILTER
# ============================================================

def remove_tiny_adjustments(
    values: EnhancementValues,
) -> EnhancementValues:

    result: Dict[str, float] = {}

    for field_name in values.__dataclass_fields__:

        value = _finite_float(
            getattr(
                values,
                field_name,
                0.0,
            )
        )

        threshold = NO_OP_THRESHOLDS.get(
            field_name,
            0.0,
        )

        if abs(value) < threshold:
            value = 0.0

        result[field_name] = value

    return EnhancementValues(
        **result
    )


# ============================================================
# SPECIAL MASK LOGIC
# ============================================================

def adjust_for_mask_type(
    values: EnhancementValues,
    name: str,
) -> EnhancementValues:

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if name == "face":

        values.contrast *= 0.55
        values.clarity *= 0.25
        values.dehaze *= 0.25
        values.saturation *= 0.45
        values.vibrance *= 0.50

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    elif name in {
        "skin",
        "skin_face",
        "skin_neck",
    }:

        values.contrast *= 0.30
        values.clarity *= 0.08
        values.dehaze *= 0.10

        values.saturation *= 0.25
        values.vibrance *= 0.20

        values.highlights *= 0.70
        values.shadows *= 0.80

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    elif name == "eyes":

        values.exposure *= 0.40
        values.contrast *= 0.50
        values.clarity *= 0.80
        values.dehaze *= 0.25

        values.saturation *= 0.30
        values.vibrance *= 0.25

    # --------------------------------------------------------
    # LIPS
    # --------------------------------------------------------

    elif name in {
        "lips",
        "mouth",
    }:

        values.contrast *= 0.45
        values.clarity *= 0.35
        values.saturation *= 0.35
        values.vibrance *= 0.30

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    elif name == "hair":

        values.contrast *= 0.65
        values.clarity *= 0.65
        values.dehaze *= 0.40

    # --------------------------------------------------------
    # CLOTHING
    # --------------------------------------------------------

    elif name in {
        "clothing",
        "top",
        "dress",
        "skirt",
        "pants",
        "torso",
    }:

        values.contrast *= 0.75
        values.clarity *= 0.75

    # --------------------------------------------------------
    # SMALL DETAILS
    # --------------------------------------------------------

    elif name in {
        "jewelry",
        "glasses",
        "brows",
    }:

        values.exposure *= 0.40
        values.contrast *= 0.45
        values.clarity *= 0.50

    return values


# ============================================================
# WHITE BALANCE SAFETY
# ============================================================

def apply_white_balance_policy(
    values: EnhancementValues,
    name: str,
    wb_confidence: float,
) -> EnhancementValues:

    wb_confidence = float(
        np.clip(
            _finite_float(
                wb_confidence,
                0.0,
            ),
            0.0,
            1.0,
        )
    )

    if name in WB_DISABLED_MASKS:

        values.temperature = 0.0
        values.tint = 0.0

        return values

    if name in WB_WEAK_MASKS:

        values.temperature *= 0.20
        values.tint *= 0.20

    elif name in WB_ALLOWED_MASKS:

        # WB must have meaningful confidence.
        values.temperature *= (
            0.25
            +
            0.75 * wb_confidence
        )

        values.tint *= (
            0.25
            +
            0.75 * wb_confidence
        )

    else:

        # Unknown masks are never trusted for WB.
        values.temperature *= 0.10
        values.tint *= 0.10

    return values


# ============================================================
# MASK QUALITY
# ============================================================

def calculate_mask_quality(
    mask: np.ndarray,
    image_shape: tuple,
    pixel_count: int,
) -> Dict[str, float]:

    height, width = image_shape[:2]

    total_pixels = max(
        height * width,
        1,
    )

    coverage = (
        pixel_count
        /
        total_pixels
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    mask = np.nan_to_num(
        mask,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mask = np.clip(
        mask,
        0.0,
        1.0,
    )

    active = mask > 0.15

    if not np.any(active):

        return {
            "coverage": 0.0,
            "soft_mean": 0.0,
            "fragmentation": 1.0,
            "edge_fraction": 1.0,
            "quality_score": 0.0,
        }

    soft_mean = float(
        np.mean(
            mask[active]
        )
    )

    # Approximate fragmentation without requiring
    # another connected-components dependency.
    ys, xs = np.where(active)

    if len(xs) > 1:

        area_box = (
            max(
                (xs.max() - xs.min() + 1),
                1,
            )
            *
            max(
                (ys.max() - ys.min() + 1),
                1,
            )
        )

        density = (
            pixel_count
            /
            max(area_box, 1)
        )

        fragmentation = float(
            np.clip(
                1.0 - density,
                0.0,
                1.0,
            )
        )

    else:

        fragmentation = 1.0

    # Boundary fraction.
    edge = np.zeros_like(
        active,
        dtype=bool,
    )

    edge[1:, :] |= (
        active[1:, :]
        !=
        active[:-1, :]
    )

    edge[:, 1:] |= (
        active[:, 1:]
        !=
        active[:, :-1]
    )

    edge_fraction = (
        float(
            np.count_nonzero(
                edge & active
            )
        )
        /
        max(
            pixel_count,
            1,
        )
    )

    coverage_score = float(
        np.clip(
            math.sqrt(
                max(
                    coverage,
                    0.0,
                )
                /
                0.02
            ),
            0.0,
            1.0,
        )
    )

    fragmentation_score = (
        1.0
        -
        fragmentation
    )

    edge_score = float(
        np.clip(
            1.0 - edge_fraction,
            0.0,
            1.0,
        )
    )

    quality_score = float(
        np.clip(
            0.35 * soft_mean
            +
            0.25 * coverage_score
            +
            0.20 * fragmentation_score
            +
            0.20 * edge_score,
            0.0,
            1.0,
        )
    )

    return {
        "coverage": float(coverage),
        "soft_mean": soft_mean,
        "fragmentation": fragmentation,
        "edge_fraction": edge_fraction,
        "quality_score": quality_score,
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    pixel_count: int,
    mask_quality: float,
    wb_confidence: float,
    region_stats: Any,
    surrounding_stats: Any,
) -> Dict[str, float]:

    pixel_confidence = float(
        np.clip(
            math.log10(
                max(
                    pixel_count,
                    1,
                )
                + 10
            )
            /
            5.0,
            0.0,
            1.0,
        )
    )

    mask_quality = float(
        np.clip(
            _finite_float(
                mask_quality,
                0.0,
            ),
            0.0,
            1.0,
        )
    )

    wb_confidence = float(
        np.clip(
            _finite_float(
                wb_confidence,
                0.0,
            ),
            0.0,
            1.0,
        )
    )

    # Statistical stability.
    stability = 0.75

    try:

        p25 = _finite_float(
            region_stats.p25
        )

        p50 = _finite_float(
            region_stats.p50
        )

        p75 = _finite_float(
            region_stats.p75
        )

        spread = abs(
            p75 - p25
        )

        # Extremely unstable regions are less trustworthy.
        stability = float(
            np.clip(
                1.0
                -
                min(
                    spread / 0.70,
                    1.0,
                ),
                0.25,
                1.0,
            )
        )

    except Exception:
        pass

    context_quality = (
        0.65
        if surrounding_stats is not None
        else 0.40
    )

    tone_confidence = float(
        np.clip(
            0.45 * pixel_confidence
            +
            0.30 * mask_quality
            +
            0.15 * stability
            +
            0.10 * context_quality,
            0.0,
            1.0,
        )
    )

    color_confidence = float(
        np.clip(
            0.35 * pixel_confidence
            +
            0.35 * mask_quality
            +
            0.20 * stability
            +
            0.10 * context_quality,
            0.0,
            1.0,
        )
    )

    confidence = float(
        np.clip(
            0.30 * pixel_confidence
            +
            0.25 * mask_quality
            +
            0.20 * stability
            +
            0.15 * context_quality
            +
            0.10 * wb_confidence,
            0.0,
            1.0,
        )
    )

    return {
        "confidence": confidence,
        "tone_confidence": tone_confidence,
        "color_confidence": color_confidence,
        "stability": stability,
    }


# ============================================================
# AUTO MASK ENHANCER
# ============================================================

class AutoMaskEnhancer:

    def __init__(
        self,
        image: np.ndarray,
        hard_masks: Dict[str, np.ndarray],
        soft_masks: Dict[str, np.ndarray],
        global_strength: float = 1.0,
    ):

        if image is None:
            raise ValueError(
                "image cannot be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "image must be numpy.ndarray."
            )

        if image.ndim != 3:
            raise ValueError(
                "image must be a 3-channel image."
            )

        self.image = image

        self.hard_masks = (
            hard_masks
            if hard_masks is not None
            else {}
        )

        self.soft_masks = (
            soft_masks
            if soft_masks is not None
            else {}
        )

        self.global_strength = float(
            np.clip(
                _finite_float(
                    global_strength,
                    1.0,
                ),
                0.0,
                2.0,
            )
        )

        logger.info(
            "Preparing color-science image data..."
        )

        self.data = prepare_color_data(
            image
        )

        self.global_stats = (
            calculate_global_stats(
                self.data
            )
        )

        logger.info(
            "Global luminance median: %.4f",
            self.global_stats.p50,
        )

        logger.info(
            "Global luminance range: %.4f - %.4f",
            self.global_stats.p05,
            self.global_stats.p95,
        )

        self._mask_cache: Dict[
            str,
            np.ndarray,
        ] = {}

    # ========================================================
    # MASK NAMES
    # ========================================================

    def mask_names(self) -> Iterable[str]:

        return sorted(
            set(
                self.hard_masks
            )
            |
            set(
                self.soft_masks
            ),
            key=lambda name: (
                -MASK_PRIORITY.get(
                    name,
                    10,
                ),
                name,
            ),
        )

    # ========================================================
    # MASK
    # ========================================================

    def get_mask(
        self,
        name: str,
    ) -> Optional[np.ndarray]:

        if name in self._mask_cache:
            return self._mask_cache[name]

        hard = self.hard_masks.get(
            name
        )

        soft = self.soft_masks.get(
            name
        )

        if hard is None and soft is None:
            return None

        if soft is not None:

            soft = np.asarray(
                soft,
                dtype=np.float32,
            )

            soft = np.nan_to_num(
                soft,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            soft = np.clip(
                soft,
                0.0,
                1.0,
            )

            if hard is not None:

                hard = np.asarray(
                    hard
                )

                if hard.shape != soft.shape:

                    logger.warning(
                        "Mask shape mismatch for %s: "
                        "hard=%s soft=%s",
                        name,
                        hard.shape,
                        soft.shape,
                    )

                    return None

                hard_f = (
                    hard > 0.15
                ).astype(
                    np.float32
                )

                # Preserve confidence inside the hard
                # segmentation instead of converting the
                # entire mask back to binary.
                mask = soft * hard_f

            else:

                mask = soft

        else:

            hard = np.asarray(
                hard
            )

            mask = (
                hard > 0.15
            ).astype(
                np.float32
            )

        if mask.shape != self.image.shape[:2]:

            logger.warning(
                "Mask %s has invalid shape: %s; "
                "expected %s",
                name,
                mask.shape,
                self.image.shape[:2],
            )

            return None

        mask = np.nan_to_num(
            mask,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        mask = np.clip(
            mask,
            0.0,
            1.0,
        )

        self._mask_cache[name] = mask

        return mask

    # ========================================================
    # CONTEXT RING
    # ========================================================

    def get_context_ring(
        self,
        name: str,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:

        try:

            ring = surrounding_ring(
                mask,
                self.image.shape[:2],
            )

        except Exception as exc:

            logger.debug(
                "Could not create context ring for %s: %s",
                name,
                exc,
            )

            return None

        if ring is None:
            return None

        ring = np.asarray(
            ring,
            dtype=np.float32,
        )

        if ring.shape != self.image.shape[:2]:
            return None

        # Remove all known semantic masks from the context.
        #
        # This prevents a face's context from being dominated
        # by another detected face or another semantic region.
        for other_name in self.mask_names():

            if other_name == name:
                continue

            other = self.get_mask(
                other_name
            )

            if other is None:
                continue

            ring[
                other > 0.25
            ] = 0.0

        ring[
            mask > 0.15
        ] = 0.0

        if np.count_nonzero(
            ring > 0.15
        ) < 20:

            return None

        return ring

    # ========================================================
    # CONTEXT MATCH
    # ========================================================

    def calculate_context_match(
        self,
        region_stats: Any,
        surrounding_stats: Any,
    ) -> Dict[str, float]:

        if surrounding_stats is None:

            return {
                "score": 0.55,
                "luminance_gap_ev": 0.0,
                "chroma_gap": 0.0,
            }

        try:

            region_luma = max(
                _finite_float(
                    region_stats.luminance_median
                ),
                EPSILON,
            )

            surrounding_luma = max(
                _finite_float(
                    surrounding_stats.luminance_median
                ),
                EPSILON,
            )

            luminance_gap = abs(
                math.log2(
                    region_luma
                    /
                    surrounding_luma
                )
            )

        except Exception:

            luminance_gap = 0.0

        try:

            chroma_gap = math.sqrt(
                (
                    _finite_float(
                        region_stats.lab_a
                    )
                    -
                    _finite_float(
                        surrounding_stats.lab_a
                    )
                ) ** 2
                +
                (
                    _finite_float(
                        region_stats.lab_b
                    )
                    -
                    _finite_float(
                        surrounding_stats.lab_b
                    )
                ) ** 2
            ) / 40.0

        except Exception:

            chroma_gap = 0.0

        luminance_component = float(
            np.exp(
                -0.70
                *
                min(
                    luminance_gap,
                    2.0,
                )
            )
        )

        chroma_component = float(
            np.exp(
                -0.45
                *
                min(
                    chroma_gap,
                    2.0,
                )
            )
        )

        score = float(
            np.clip(
                0.60 * luminance_component
                +
                0.40 * chroma_component,
                0.0,
                1.0,
            )
        )

        return {
            "score": score,
            "luminance_gap_ev": float(
                luminance_gap
            ),
            "chroma_gap": float(
                chroma_gap
            ),
        }

    # ========================================================
    # ANALYZE MASK
    # ========================================================

    def analyze_mask(
        self,
        name: str,
    ) -> Optional[Dict[str, Any]]:

        mask = self.get_mask(
            name
        )

        if mask is None:
            return None

        pixel_count = int(
            np.count_nonzero(
                mask > 0.15
            )
        )

        if pixel_count == 0:
            return None

        minimum = int(
            MIN_PIXELS.get(
                name,
                100,
            )
        )

        if pixel_count < minimum:

            logger.debug(
                "Mask %s is too small: %d pixels",
                name,
                pixel_count,
            )

            return {
                "enabled": False,
                "reason": "mask_too_small",
                "pixel_count": pixel_count,
                "adjustments": asdict(
                    EnhancementValues()
                ),
            }

        # ----------------------------------------------------
        # MASK QUALITY
        # ----------------------------------------------------

        quality = calculate_mask_quality(
            mask,
            self.image.shape,
            pixel_count,
        )

        mask_quality = quality[
            "quality_score"
        ]

        if mask_quality < 0.15:

            return {
                "enabled": False,
                "reason": "mask_quality_too_low",
                "pixel_count": pixel_count,
                "mask_quality": quality,
                "adjustments": asdict(
                    EnhancementValues()
                ),
            }

        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        region_stats = calculate_region_stats(
            self.data,
            mask,
        )

        if region_stats is None:
            return None

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        ring = self.get_context_ring(
            name,
            mask,
        )

        surrounding_stats = None

        if ring is not None:

            surrounding_stats = (
                calculate_region_stats(
                    self.data,
                    ring,
                )
            )

        # ----------------------------------------------------
        # WHITE BALANCE
        # ----------------------------------------------------

        region_wb = estimate_white_balance(
            self.data,
            mask,
        )

        surrounding_wb = None

        if surrounding_stats is not None:

            surrounding_wb = (
                estimate_white_balance(
                    self.data,
                    ring,
                )
            )

        wb_confidence = _finite_float(
            getattr(
                region_wb,
                "confidence",
                0.0,
            )
        )

        wb_confidence = float(
            np.clip(
                wb_confidence,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # EXPOSURE
        # ----------------------------------------------------

        exposure = exposure_from_luminance(
            region_stats
        )

        exposure += (
            context_exposure_adjustment(
                region_stats,
                surrounding_stats,
            )
        )

        # ----------------------------------------------------
        # TONE
        # ----------------------------------------------------

        tone = calculate_tone_adjustments(
            region_stats
        )

        if tone is None:
            tone = {}

        # ----------------------------------------------------
        # CONTRAST
        # ----------------------------------------------------

        contrast = calculate_contrast(
            region_stats
        )

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        color = calculate_color_adjustments(
            self.data,
            mask,
            name,
        )

        if color is None:
            color = {}

        # ----------------------------------------------------
        # CLARITY
        # ----------------------------------------------------

        clarity = calculate_clarity(
            self.data,
            mask,
            name,
        )

        # ----------------------------------------------------
        # DEHAZE
        # ----------------------------------------------------

        dehaze = calculate_dehaze(
            region_stats
        )

        # ----------------------------------------------------
        # WHITE BALANCE
        # ----------------------------------------------------

        temperature = 0.0
        tint = 0.0

        if name in WB_ALLOWED_MASKS:

            temperature, tint = (
                context_white_balance(
                    region_wb,
                    surrounding_wb,
                    name,
                )
            )

        # ----------------------------------------------------
        # ASSEMBLE
        # ----------------------------------------------------

        values = EnhancementValues(

            exposure=_finite_float(
                exposure
            ),

            contrast=_finite_float(
                contrast
            ),

            highlights=_finite_float(
                tone.get(
                    "highlights",
                    0.0,
                )
            ),

            shadows=_finite_float(
                tone.get(
                    "shadows",
                    0.0,
                )
            ),

            whites=_finite_float(
                tone.get(
                    "whites",
                    0.0,
                )
            ),

            blacks=_finite_float(
                tone.get(
                    "blacks",
                    0.0,
                )
            ),

            temperature=_finite_float(
                temperature
            ),

            tint=_finite_float(
                tint
            ),

            saturation=_finite_float(
                color.get(
                    "saturation",
                    0.0,
                )
            ),

            vibrance=_finite_float(
                color.get(
                    "vibrance",
                    0.0,
                )
            ),

            clarity=_finite_float(
                clarity
            ),

            dehaze=_finite_float(
                dehaze
            ),
        )

        # ----------------------------------------------------
        # MASK-SPECIFIC SAFETY
        # ----------------------------------------------------

        values = adjust_for_mask_type(
            values,
            name,
        )

        # ----------------------------------------------------
        # WB POLICY
        # ----------------------------------------------------

        values = apply_white_balance_policy(
            values,
            name,
            wb_confidence,
        )

        # ----------------------------------------------------
        # CONTEXT MATCH
        # ----------------------------------------------------

        context = self.calculate_context_match(
            region_stats,
            surrounding_stats,
        )

        context_match = context[
            "score"
        ]

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_data = (
            calculate_confidence(
                pixel_count,
                mask_quality,
                wb_confidence,
                region_stats,
                surrounding_stats,
            )
        )

        confidence = confidence_data[
            "confidence"
        ]

        tone_confidence = (
            confidence_data[
                "tone_confidence"
            ]
        )

        color_confidence = (
            confidence_data[
                "color_confidence"
            ]
        )

        # ----------------------------------------------------
        # CONTEXT SAFETY
        # ----------------------------------------------------

        safety_scale = float(
            np.clip(
                0.45
                +
                0.55 * context_match,
                0.35,
                1.0,
            )
        )

        values = apply_strength(
            values,
            safety_scale,
        )

        # ----------------------------------------------------
        # MASK STRENGTH
        # ----------------------------------------------------

        mask_strength = _finite_float(
            MASK_STRENGTH.get(
                name,
                DEFAULT_MASK_STRENGTH,
            ),
            DEFAULT_MASK_STRENGTH,
        )

        mask_strength = float(
            np.clip(
                mask_strength,
                0.0,
                1.0,
            )
        )

        values = apply_strength(
            values,
            mask_strength
            *
            self.global_strength,
        )

        # ----------------------------------------------------
        # CONFIDENCE STRENGTH
        #
        # Don't allow uncertain analysis to produce a full
        # strength correction.
        # ----------------------------------------------------

        confidence_scale = float(
            np.clip(
                0.35
                +
                0.65 * confidence,
                0.35,
                1.0,
            )
        )

        values = apply_strength(
            values,
            confidence_scale,
        )

        # ----------------------------------------------------
        # EDIT BUDGET
        # ----------------------------------------------------

        values = apply_edit_budget(
            values,
            name,
            confidence,
        )

        # ----------------------------------------------------
        # GLOBAL LIMITS
        # ----------------------------------------------------

        values = limit_values(
            values
        )

        # ----------------------------------------------------
        # REMOVE TINY CHANGES
        # ----------------------------------------------------

        values = remove_tiny_adjustments(
            values
        )

        # ----------------------------------------------------
        # SHOULD APPLY
        # ----------------------------------------------------

        nonzero_count = 0

        for field_name in values.__dataclass_fields__:

            value = _finite_float(
                getattr(
                    values,
                    field_name,
                    0.0,
                )
            )

            threshold = NO_OP_THRESHOLDS.get(
                field_name,
                0.0,
            )

            if abs(value) >= threshold:
                nonzero_count += 1

        should_apply = (
            nonzero_count > 0
            and
            confidence >= 0.20
            and
            mask_quality >= 0.15
        )

        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        coverage = (
            pixel_count
            /
            max(
                self.image.shape[0]
                *
                self.image.shape[1],
                1,
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "enabled": True,

            "pixel_count": pixel_count,

            "coverage": float(
                coverage
            ),

            "confidence": float(
                confidence
            ),

            "mask_quality": {
                "score": float(
                    mask_quality
                ),
                "soft_mean": float(
                    quality["soft_mean"]
                ),
                "fragmentation": float(
                    quality["fragmentation"]
                ),
                "edge_fraction": float(
                    quality["edge_fraction"]
                ),
            },

            "context_match": {
                "score": float(
                    context_match
                ),

                "luminance_gap_ev": float(
                    context[
                        "luminance_gap_ev"
                    ]
                ),

                "chroma_gap": float(
                    context[
                        "chroma_gap"
                    ]
                ),

                "safety_scale": float(
                    safety_scale
                ),
            },

            "confidence_detail": {
                "overall": float(
                    confidence
                ),

                "tone": float(
                    tone_confidence
                ),

                "color": float(
                    color_confidence
                ),

                "white_balance": float(
                    wb_confidence
                ),

                "stability": float(
                    confidence_data[
                        "stability"
                    ]
                ),
            },

            "priority": int(
                MASK_PRIORITY.get(
                    name,
                    10,
                )
            ),

            "white_balance_policy": (
                "allowed"
                if name in WB_ALLOWED_MASKS
                else
                "weak"
                if name in WB_WEAK_MASKS
                else
                "disabled"
            ),

            "should_apply": bool(
                should_apply
            ),

            "analysis": {

                "luminance": {
                    "mean": _finite_float(
                        region_stats.luminance_mean
                    ),

                    "median": _finite_float(
                        region_stats.luminance_median
                    ),

                    "p01": _finite_float(
                        region_stats.p01
                    ),

                    "p05": _finite_float(
                        region_stats.p05
                    ),

                    "p10": _finite_float(
                        region_stats.p10
                    ),

                    "p25": _finite_float(
                        region_stats.p25
                    ),

                    "p50": _finite_float(
                        region_stats.p50
                    ),

                    "p75": _finite_float(
                        region_stats.p75
                    ),

                    "p90": _finite_float(
                        region_stats.p90
                    ),

                    "p95": _finite_float(
                        region_stats.p95
                    ),

                    "p99": _finite_float(
                        region_stats.p99
                    ),
                },

                "contrast": {
                    "rms": _finite_float(
                        region_stats.rms_contrast
                    ),

                    "dynamic_range": (
                        _finite_float(
                            region_stats.p95
                        )
                        -
                        _finite_float(
                            region_stats.p05
                        )
                    ),
                },

                "clipping": {
                    "shadow_fraction": (
                        _finite_float(
                            region_stats.shadow_fraction
                        )
                    ),

                    "highlight_fraction": (
                        _finite_float(
                            region_stats.highlight_fraction
                        )
                    ),
                },

                "color": {
                    "lab_L": _finite_float(
                        region_stats.lab_L
                    ),

                    "lab_a": _finite_float(
                        region_stats.lab_a
                    ),

                    "lab_b": _finite_float(
                        region_stats.lab_b
                    ),

                    "chroma": _finite_float(
                        region_stats.chroma
                    ),

                    "red_green_ratio": (
                        _finite_float(
                            region_stats.red_green_ratio
                        )
                    ),

                    "blue_yellow_ratio": (
                        _finite_float(
                            region_stats.blue_yellow_ratio
                        )
                    ),

                    "saturation": _finite_float(
                        region_stats.saturation
                    ),
                },

                "white_balance": {
                    "temperature_estimate": (
                        _finite_float(
                            getattr(
                                region_wb,
                                "temperature",
                                0.0,
                            )
                        )
                    ),

                    "tint_estimate": (
                        _finite_float(
                            getattr(
                                region_wb,
                                "tint",
                                0.0,
                            )
                        ),

                    ),

                    "confidence": float(
                        wb_confidence
                    ),
                },

                "surrounding": (
                    {
                        "luminance_median": (
                            _finite_float(
                                surrounding_stats.luminance_median
                            )
                        ),

                        "luminance_mean": (
                            _finite_float(
                                surrounding_stats.luminance_mean
                            )
                        ),

                        "lab_a": (
                            _finite_float(
                                surrounding_stats.lab_a
                            )
                        ),

                        "lab_b": (
                            _finite_float(
                                surrounding_stats.lab_b
                            )
                        ),
                    }
                    if surrounding_stats is not None
                    else None
                ),
            },

            "adjustments": asdict(
                values
            ),
        }

    # ========================================================
    # ANALYZE ALL MASKS
    # ========================================================

    def analyze(
        self,
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {}

        # IMPORTANT:
        # Use the union of hard and soft masks.
        mask_names = list(
            self.mask_names()
        )

        for name in mask_names:

            if name in {
                "background",
                "non_person",
            }:
                continue

            logger.info(
                "Analyzing mask: %s",
                name,
            )

            try:

                result = self.analyze_mask(
                    name
                )

            except Exception as exc:

                logger.exception(
                    "Failed to analyze mask %s",
                    name,
                )

                result = {
                    "enabled": False,
                    "reason": "analysis_error",
                    "error": str(exc),
                    "adjustments": asdict(
                        EnhancementValues()
                    ),
                }

            if result is not None:

                results[name] = result

        # ----------------------------------------------------
        # OVERLAP INFORMATION
        #
        # The actual per-pixel conflict resolution should be
        # performed by the enhancement stage, but the analyzer
        # exposes priority information for that stage.
        # ----------------------------------------------------

        ordered_masks = sorted(
            results.keys(),
            key=lambda n: -MASK_PRIORITY.get(
                n,
                10,
            ),
        )

        results["_meta"] = {
            "mask_order": ordered_masks,

            "mask_priority": {
                name: MASK_PRIORITY.get(
                    name,
                    10,
                )
                for name in ordered_masks
            },

            "overlap_policy": (
                "higher_priority_masks_override_"
                "lower_priority_masks"
            ),
        }

        return results