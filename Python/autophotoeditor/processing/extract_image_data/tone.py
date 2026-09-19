from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np

from .shared import *


# ============================================================================
# EXPOSURE CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ExposureConfig:
    """
    Conservative photographic exposure configuration.

    Values are intentionally ranges rather than a single "correct" exposure.
    """

    min_ev: float = -1.50
    max_ev: float = 1.50

    coarse_step: float = 0.10
    fine_step: float = 0.02

    # Neutral photographic midtone region.
    target_p50_min: float = 0.16
    target_p50_max: float = 0.25

    # Subject protection.
    subject_target_min: float = 0.18
    subject_target_max: float = 0.62

    # Highlight protection.
    highlight_soft_limit: float = 0.92
    highlight_hard_limit: float = 0.985
    highlight_clip_limit: float = 0.005

    # Shadow protection.
    shadow_soft_limit: float = 0.025
    shadow_hard_limit: float = 0.004
    shadow_clip_limit: float = 0.02

    # Don't make corrections unless the evidence is meaningful.
    minimum_ev_change: float = 0.035

    # Strong preference for smaller changes.
    movement_weight: float = 0.80


DEFAULT_EXPOSURE_CONFIG = ExposureConfig()


# ============================================================================
# EXPOSURE TRANSFORM
# ============================================================================

def apply_exposure_linear(
    y: np.ndarray,
    ev: float,
) -> np.ndarray:
    """
    Apply exposure in linear-light space.

    IMPORTANT:
    The result is intentionally NOT clipped.

    Clipping during optimization destroys information about how much a
    candidate exposure would actually exceed the display range.
    """

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    gain = float(
        2.0 ** float(ev)
    )

    return np.maximum(
        y * gain,
        0.0,
    )


# ============================================================================
# EXPOSURE MEASUREMENTS
# ============================================================================

@dataclass(frozen=True)
class ExposureMetrics:
    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    mean: float
    std: float

    shadow_clip: float
    highlight_clip: float

    shadow_headroom: float
    highlight_headroom: float

    dynamic_range: float


def measure_exposure(
    y: np.ndarray,
) -> ExposureMetrics:

    values = np.asarray(
        y,
        dtype=np.float32,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return ExposureMetrics(
            p01=0.0,
            p05=0.0,
            p10=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            mean=0.0,
            std=0.0,
            shadow_clip=0.0,
            highlight_clip=0.0,
            shadow_headroom=0.0,
            highlight_headroom=0.0,
            dynamic_range=0.0,
        )

    values = sample_array(
        values,
        MAX_OPTIMIZATION_PIXELS,
    )

    values = np.maximum(
        values,
        0.0,
    )

    p01 = percentile(values, 1)
    p05 = percentile(values, 5)
    p10 = percentile(values, 10)
    p25 = percentile(values, 25)
    p50 = percentile(values, 50)
    p75 = percentile(values, 75)
    p90 = percentile(values, 90)
    p95 = percentile(values, 95)
    p99 = percentile(values, 99)

    # Don't use only clipped 0..1 values here.
    # Values > 1 contain useful information about highlight risk.
    shadow_clip = float(
        np.mean(values <= 0.003)
    )

    highlight_clip = float(
        np.mean(values >= 0.995)
    )

    # Headroom is measured from the actual percentile.
    highlight_headroom = max(
        0.0,
        1.0 - p95,
    )

    shadow_headroom = max(
        0.0,
        p05,
    )

    dynamic_range = max(
        0.0,
        p95 - p05,
    )

    return ExposureMetrics(
        p01=float(p01),
        p05=float(p05),
        p10=float(p10),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),

        mean=float(
            robust_mean(values)
        ),

        std=float(
            robust_std(values)
        ),

        shadow_clip=shadow_clip,
        highlight_clip=highlight_clip,

        shadow_headroom=float(
            shadow_headroom
        ),

        highlight_headroom=float(
            highlight_headroom
        ),

        dynamic_range=float(
            dynamic_range
        ),
    )


# ============================================================================
# SCENE TARGETS
# ============================================================================

@dataclass(frozen=True)
class ExposureTarget:
    p50_min: float
    p50_max: float

    subject_min: float
    subject_max: float

    allow_dark_bias: bool = False
    allow_bright_bias: bool = False


def get_exposure_target(
    scene_type: str,
) -> ExposureTarget:
    """
    Return a RANGE rather than a single target.

    This is crucial.

    Auto Tone should not force every photograph toward exactly the same
    luminance distribution.
    """

    scene = (
        scene_type or "normal"
    ).lower()

    if scene == "low_key":
        return ExposureTarget(
            p50_min=0.09,
            p50_max=0.19,
            subject_min=0.16,
            subject_max=0.55,
            allow_dark_bias=True,
        )

    if scene == "high_key":
        return ExposureTarget(
            p50_min=0.20,
            p50_max=0.34,
            subject_min=0.25,
            subject_max=0.70,
            allow_bright_bias=True,
        )

    if scene == "backlit":
        return ExposureTarget(
            p50_min=0.14,
            p50_max=0.28,
            subject_min=0.24,
            subject_max=0.62,
        )

    if scene == "portrait":
        return ExposureTarget(
            p50_min=0.15,
            p50_max=0.28,
            subject_min=0.24,
            subject_max=0.65,
        )

    if scene == "landscape":
        return ExposureTarget(
            p50_min=0.14,
            p50_max=0.28,
            subject_min=0.18,
            subject_max=0.62,
        )

    return ExposureTarget(
        p50_min=0.14,
        p50_max=0.27,
        subject_min=0.18,
        subject_max=0.62,
    )


# ============================================================================
# RANGE PENALTY
# ============================================================================

def range_penalty(
    value: float,
    low: float,
    high: float,
    weight: float,
) -> float:

    if value < low:
        distance = low - value
        return distance * distance * weight

    if value > high:
        distance = value - high
        return distance * distance * weight

    return 0.0


# ============================================================================
# EXPOSURE CANDIDATE
# ============================================================================

def score_exposure_candidate(
    y: np.ndarray,
    subject_y: Optional[np.ndarray],
    ev: float,
    target: ExposureTarget,
    config: ExposureConfig,
) -> Tuple[float, Dict[str, float]]:

    transformed = apply_exposure_linear(
        y,
        ev,
    )

    metrics = measure_exposure(
        transformed
    )

    # ------------------------------------------------------------
    # Global midtone objective
    # ------------------------------------------------------------

    median_penalty = range_penalty(
        metrics.p50,
        target.p50_min,
        target.p50_max,
        weight=10.0,
    )

    # ------------------------------------------------------------
    # Highlight preservation
    # ------------------------------------------------------------

    highlight_soft = max(
        0.0,
        metrics.p95
        - config.highlight_soft_limit,
    )

    highlight_hard = max(
        0.0,
        metrics.p99
        - config.highlight_hard_limit,
    )

    highlight_clip = max(
        0.0,
        metrics.highlight_clip
        - config.highlight_clip_limit,
    )

    highlight_penalty = (
        highlight_soft ** 2 * 18.0
        + highlight_hard ** 2 * 80.0
        + highlight_clip * 15.0
    )

    # ------------------------------------------------------------
    # Shadow preservation
    # ------------------------------------------------------------

    shadow_soft = max(
        0.0,
        config.shadow_soft_limit
        - metrics.p05,
    )

    shadow_hard = max(
        0.0,
        config.shadow_hard_limit
        - metrics.p01,
    )

    shadow_clip = max(
        0.0,
        metrics.shadow_clip
        - config.shadow_clip_limit,
    )

    shadow_penalty = (
        shadow_soft ** 2 * 16.0
        + shadow_hard ** 2 * 12.0
        + shadow_clip * 4.0
    )

    # ------------------------------------------------------------
    # Subject
    # ------------------------------------------------------------

    subject_penalty = 0.0

    if (
        subject_y is not None
        and subject_y.size >= 100
    ):

        subject_transformed = apply_exposure_linear(
            subject_y,
            ev,
        )

        subject_metrics = measure_exposure(
            subject_transformed
        )

        subject_penalty += range_penalty(
            subject_metrics.p50,
            target.subject_min,
            target.subject_max,
            weight=13.0,
        )

        # Subject highlight protection is stronger than global protection.
        subject_highlight = max(
            0.0,
            subject_metrics.p95
            - 0.93,
        )

        subject_clip = max(
            0.0,
            subject_metrics.highlight_clip
            - 0.003,
        )

        subject_penalty += (
            subject_highlight ** 2 * 28.0
            + subject_clip * 30.0
        )

    # ------------------------------------------------------------
    # Dynamic-range preservation
    # ------------------------------------------------------------

    # Don't reward flattening the image simply to hit the median target.
    dynamic_penalty = 0.0

    if metrics.dynamic_range < 0.20:
        dynamic_penalty = (
            0.20
            - metrics.dynamic_range
        ) ** 2 * 5.0

    # ------------------------------------------------------------
    # Movement penalty
    # ------------------------------------------------------------

    movement_penalty = (
        ev * ev
        * config.movement_weight
    )

    # ------------------------------------------------------------
    # Directional penalty
    # ------------------------------------------------------------

    # When an image already has substantial highlight pressure,
    # increasing exposure should become increasingly expensive.

    directional_penalty = 0.0

    if ev > 0.0:

        highlight_pressure = clamp(
            metrics.p95,
            0.0,
            1.0,
        )

        directional_penalty += (
            ev
            * highlight_pressure
            * 1.5
        )

    if ev < 0.0:

        shadow_pressure = clamp(
            1.0 - metrics.p05 * 8.0,
            0.0,
            1.0,
        )

        directional_penalty += (
            abs(ev)
            * shadow_pressure
            * 1.2
        )

    total = (
        median_penalty
        + highlight_penalty
        + shadow_penalty
        + subject_penalty
        + dynamic_penalty
        + movement_penalty
        + directional_penalty
    )

    return total, {
        "median": float(median_penalty),
        "highlights": float(highlight_penalty),
        "shadows": float(shadow_penalty),
        "subject": float(subject_penalty),
        "dynamic_range": float(dynamic_penalty),
        "movement": float(movement_penalty),
        "directional": float(directional_penalty),
    }


# ============================================================================
# SUBJECT EXTRACTION
# ============================================================================

def extract_subject_luminance(
    y: np.ndarray,
    subject_mask: Optional[np.ndarray],
    max_pixels: int = 80_000,
) -> Optional[np.ndarray]:

    if subject_mask is None:
        return None

    mask = normalized_mask(
        subject_mask
    )

    if mask is None:
        return None

    if mask.shape != y.shape[:2]:
        return None

    values = y[mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size < 100:
        return None

    return sample_array(
        values,
        max_pixels,
    )


# ============================================================================
# EXPOSURE SEARCH
# ============================================================================

def find_best_exposure_ev(
    y: np.ndarray,
    subject_y: Optional[np.ndarray],
    target: ExposureTarget,
    config: ExposureConfig,
) -> Tuple[float, float, Dict[str, float]]:

    best_ev = 0.0
    best_score = float("inf")
    best_details: Dict[str, float] = {}

    # ------------------------------------------------------------
    # Coarse search
    # ------------------------------------------------------------

    coarse_values = np.arange(
        config.min_ev,
        config.max_ev + 0.0001,
        config.coarse_step,
        dtype=np.float32,
    )

    for ev in coarse_values:

        score, details = score_exposure_candidate(
            y,
            subject_y,
            float(ev),
            target,
            config,
        )

        if score < best_score:

            best_score = score
            best_ev = float(ev)
            best_details = details

    # ------------------------------------------------------------
    # Fine search
    # ------------------------------------------------------------

    fine_min = max(
        config.min_ev,
        best_ev - 0.12,
    )

    fine_max = min(
        config.max_ev,
        best_ev + 0.12,
    )

    fine_values = np.arange(
        fine_min,
        fine_max + 0.0001,
        config.fine_step,
        dtype=np.float32,
    )

    for ev in fine_values:

        score, details = score_exposure_candidate(
            y,
            subject_y,
            float(ev),
            target,
            config,
        )

        if score < best_score:

            best_score = score
            best_ev = float(ev)
            best_details = details

    return (
        best_ev,
        best_score,
        best_details,
    )


# ============================================================================
# EXPOSURE CONFIDENCE
# ============================================================================

def calculate_exposure_confidence(
    zero_score: float,
    best_score: float,
    original: ExposureMetrics,
    adjusted: ExposureMetrics,
    ev: float,
) -> float:

    if zero_score <= EPS:
        return 0.15

    improvement = (
        zero_score - best_score
    ) / max(
        zero_score,
        EPS,
    )

    improvement_confidence = clamp(
        improvement * 3.0,
        0.0,
        1.0,
    )

    # Strong confidence only when the original image actually has
    # measurable tonal problems.
    problem_strength = 0.0

    problem_strength += clamp(
        (0.015 - original.shadow_clip)
        * 20.0,
        0.0,
        1.0,
    )

    problem_strength += clamp(
        (original.highlight_clip - 0.01)
        * 20.0,
        0.0,
        1.0,
    )

    problem_strength += clamp(
        abs(
            original.p50
            - 0.20
        ) * 3.0,
        0.0,
        1.0,
    )

    problem_strength /= 3.0

    movement_confidence = clamp(
        abs(ev) / 0.75,
        0.0,
        1.0,
    )

    confidence = (
        improvement_confidence * 0.45
        + problem_strength * 0.40
        + movement_confidence * 0.15
    )

    # If the proposed correction makes clipping materially worse,
    # confidence must be reduced.
    if (
        adjusted.highlight_clip
        > original.highlight_clip + 0.01
    ):
        confidence *= 0.45

    if (
        adjusted.shadow_clip
        > original.shadow_clip + 0.03
    ):
        confidence *= 0.60

    return clamp(
        confidence,
        0.05,
        0.98,
    )


# ============================================================================
# PUBLIC EXPOSURE ANALYZER
# ============================================================================

def derive_lightroom_tone_controls(
    y_linear: np.ndarray,
    exposure_ev: float,
    scene_type: str,
    subject_mask: Optional[np.ndarray] = None,
    config: ExposureConfig = DEFAULT_EXPOSURE_CONFIG,
) -> Dict[str, float]:
    """
    Generate conservative Lightroom-style light controls from an exposure
    recommendation and the current luminance distribution.

    The function intentionally stays in a small range so neutral images remain
    essentially unchanged while still providing useful edit guidance for more
    extreme scenes.
    """

    y = np.asarray(
        y_linear,
        dtype=np.float32,
    )

    if y.ndim != 2:
        raise ValueError(
            "y_linear must be a 2D luminance array."
        )

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    y = np.clip(
        y,
        0.0,
        1.0,
    )

    stats = measure_exposure(
        sample_array(
            y,
            MAX_OPTIMIZATION_PIXELS,
        ).reshape(-1)
    )

    ev = float(
        safe_float(
            exposure_ev,
            0.0,
        )
    )

    if abs(ev) < config.minimum_ev_change:
        ev = 0.0

    median = stats.p50
    shadow_clip = stats.shadow_clip
    highlight_clip = stats.highlight_clip
    tonal_width = max(0.0, stats.p90 - stats.p10)

    contrast = clamp(
        (median - 0.18) * 2.5 + ev * 0.25,
        -1.5,
        1.5,
    )

    highlights = clamp(
        (highlight_clip - 0.012) * 26.0 - ev * 0.30,
        -1.5,
        1.5,
    )

    shadows = clamp(
        (shadow_clip - 0.035) * 26.0 + ev * 0.35,
        -1.5,
        1.5,
    )

    whites = clamp(
        (stats.p95 - 0.72) * 5.0 + ev * 0.20,
        -1.5,
        1.5,
    )

    blacks = clamp(
        (0.20 - stats.p05) * 6.0 - ev * 0.20,
        -1.5,
        1.5,
    )

    if scene_type.lower() in {"night", "low_light", "dark"}:
        shadows = clamp(shadows * 1.25, -1.5, 1.5)
        blacks = clamp(blacks * 1.15, -1.5, 1.5)
    elif scene_type.lower() in {"snow", "beach", "high_key", "bright"}:
        highlights = clamp(highlights * 1.25, -1.5, 1.5)
        whites = clamp(whites * 1.15, -1.5, 1.5)

    return {
        "contrast": round(
            float(contrast),
            4,
        ),
        "highlights": round(
            float(highlights),
            4,
        ),
        "shadows": round(
            float(shadows),
            4,
        ),
        "whites": round(
            float(whites),
            4,
        ),
        "blacks": round(
            float(blacks),
            4,
        ),
    }


def derive_exposure_recommendation(
    y_linear: np.ndarray,
    subject_mask: Optional[np.ndarray],
    scene_type: str,
    config: ExposureConfig = DEFAULT_EXPOSURE_CONFIG,
) -> Dict[str, Any]:

    y = np.asarray(
        y_linear,
        dtype=np.float32,
    )

    if y.ndim != 2:
        raise ValueError(
            "y_linear must be a 2D luminance array."
        )

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=4.0,
        neginf=0.0,
    )

    y = np.maximum(
        y,
        0.0,
    )

    # ------------------------------------------------------------
    # Sample once
    # ------------------------------------------------------------

    y_sample = sample_array(
        y,
        MAX_OPTIMIZATION_PIXELS,
    )

    y_sample = np.asarray(
        y_sample,
        dtype=np.float32,
    ).reshape(-1)

    subject_sample = extract_subject_luminance(
        y,
        subject_mask,
    )

    # ------------------------------------------------------------
    # Original measurements
    # ------------------------------------------------------------

    original = measure_exposure(
        y_sample
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    target = get_exposure_target(
        scene_type
    )

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------

    best_ev, best_score, details = (
        find_best_exposure_ev(
            y_sample,
            subject_sample,
            target,
            config,
        )
    )

    # ------------------------------------------------------------
    # Compare against zero EV
    # ------------------------------------------------------------

    zero_score, zero_details = (
        score_exposure_candidate(
            y_sample,
            subject_sample,
            0.0,
            target,
            config,
        )
    )

    adjusted_y = apply_exposure_linear(
        y_sample,
        best_ev,
    )

    adjusted = measure_exposure(
        adjusted_y
    )

    confidence = calculate_exposure_confidence(
        zero_score,
        best_score,
        original,
        adjusted,
        best_ev,
    )

    # ------------------------------------------------------------
    # Dead zone
    # ------------------------------------------------------------

    if abs(best_ev) < config.minimum_ev_change:

        best_ev = 0.0

        adjusted = original

        confidence = min(
            confidence,
            0.15,
        )

        decision = "hold"

    else:
        decision = "adjust"

    # ------------------------------------------------------------
    # If correction does not meaningfully improve the image,
    # do nothing.
    # ------------------------------------------------------------

    improvement = (
        zero_score - best_score
    ) / max(
        zero_score,
        EPS,
    )

    if improvement < 0.025:

        best_ev = 0.0
        adjusted = original

        confidence = min(
            confidence,
            0.20,
        )

        decision = "hold"

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------

    return {
        "decision": decision,

        "ev": round(
            float(
                clamp(
                    best_ev,
                    config.min_ev,
                    config.max_ev,
                )
            ),
            4,
        ),

        "confidence": round(
            float(confidence),
            4,
        ),

        "scene": scene_type,

        "target": {
            "p50_min": target.p50_min,
            "p50_max": target.p50_max,
            "subject_min": target.subject_min,
            "subject_max": target.subject_max,
        },

        "original": {
            "p01": round(original.p01, 6),
            "p05": round(original.p05, 6),
            "p10": round(original.p10, 6),
            "p25": round(original.p25, 6),
            "p50": round(original.p50, 6),
            "p75": round(original.p75, 6),
            "p90": round(original.p90, 6),
            "p95": round(original.p95, 6),
            "p99": round(original.p99, 6),

            "mean": round(original.mean, 6),
            "std": round(original.std, 6),

            "shadow_clip": round(
                original.shadow_clip,
                6,
            ),

            "highlight_clip": round(
                original.highlight_clip,
                6,
            ),

            "dynamic_range": round(
                original.dynamic_range,
                6,
            ),
        },

        "predicted": {
            "p01": round(adjusted.p01, 6),
            "p05": round(adjusted.p05, 6),
            "p10": round(adjusted.p10, 6),
            "p25": round(adjusted.p25, 6),
            "p50": round(adjusted.p50, 6),
            "p75": round(adjusted.p75, 6),
            "p90": round(adjusted.p90, 6),
            "p95": round(adjusted.p95, 6),
            "p99": round(adjusted.p99, 6),

            "shadow_clip": round(
                adjusted.shadow_clip,
                6,
            ),

            "highlight_clip": round(
                adjusted.highlight_clip,
                6,
            ),

            "dynamic_range": round(
                adjusted.dynamic_range,
                6,
            ),
        },

        "objective": {
            "initial_score": round(
                zero_score,
                6,
            ),

            "best_score": round(
                best_score,
                6,
            ),

            "improvement": round(
                float(improvement),
                6,
            ),

            "components": {
                key: round(
                    float(value),
                    6,
                )
                for key, value in details.items()
            },

            "zero_ev_components": {
                key: round(
                    float(value),
                    6,
                )
                for key, value in zero_details.items()
            },
        },
    }