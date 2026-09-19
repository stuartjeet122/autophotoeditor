from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .shared import *


# ============================================================================
# SCENE CLASSIFICATION
# ============================================================================

def classify_scene(
    l_star: np.ndarray,
    semantic: Mapping[str, Optional[np.ndarray]],
    subject: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Classify the tonal character of an image.

    Important:
        This function describes the image.
        It must NOT decide whether the image is correctly exposed.

    Scene classes:
        - low_key
        - high_key
        - normal
        - contrasty
        - flat
        - backlit
        - unknown

    The classifier uses:

        1. L* percentiles
        2. shadow/highlight occupancy
        3. clipping
        4. tonal spread
        5. subject luminance when available
        6. background-vs-subject relationship

    This is deliberately conservative. An image should only be classified
    as low-key/high-key when multiple signals agree.
    """

    if l_star is None:
        return {
            "type": "unknown",
            "confidence": 0.0,
            "reason": "missing_luminance",
        }

    l = np.asarray(
        l_star,
        dtype=np.float32,
    )

    if l.ndim != 2 or l.size == 0:
        return {
            "type": "unknown",
            "confidence": 0.0,
            "reason": "invalid_luminance",
        }

    # ------------------------------------------------------------------
    # Deterministic sampling
    # ------------------------------------------------------------------

    values = sample_array(
        l,
        100_000,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if values.size < 32:
        return {
            "type": "unknown",
            "confidence": 0.0,
            "reason": "insufficient_pixels",
        }

    # L* is nominally 0..100.
    values = np.clip(
        values,
        0.0,
        100.0,
    )

    # ------------------------------------------------------------------
    # Tonal distribution
    # ------------------------------------------------------------------

    p01 = percentile(values, 1)
    p05 = percentile(values, 5)
    p10 = percentile(values, 10)
    p25 = percentile(values, 25)
    p50 = percentile(values, 50)
    p75 = percentile(values, 75)
    p90 = percentile(values, 90)
    p95 = percentile(values, 95)
    p99 = percentile(values, 99)

    tonal_range = p95 - p05
    iqr = p75 - p25

    # ------------------------------------------------------------------
    # Shadow / highlight occupancy
    # ------------------------------------------------------------------

    shadow_ratio = float(
        np.mean(values < 20.0)
    )

    deep_shadow_ratio = float(
        np.mean(values < 8.0)
    )

    highlight_ratio = float(
        np.mean(values > 80.0)
    )

    bright_highlight_ratio = float(
        np.mean(values > 92.0)
    )

    near_black_ratio = float(
        np.mean(values <= 2.0)
    )

    near_white_ratio = float(
        np.mean(values >= 98.0)
    )

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    black_clip_ratio = float(
        np.mean(values <= 1.0)
    )

    white_clip_ratio = float(
        np.mean(values >= 99.0)
    )

    # ------------------------------------------------------------------
    # Distribution asymmetry
    # ------------------------------------------------------------------

    lower_span = p50 - p10
    upper_span = p90 - p50

    if tonal_range > EPS:
        skew_indicator = (
            upper_span - lower_span
        ) / tonal_range
    else:
        skew_indicator = 0.0

    # ------------------------------------------------------------------
    # Subject analysis
    # ------------------------------------------------------------------

    subject_median = None
    subject_p25 = None
    subject_p75 = None
    subject_valid = False

    if subject is not None:

        subject_mask = normalized_mask(
            subject
        )

        if (
            subject_mask is not None
            and subject_mask.shape == l.shape
        ):
            subject_values = l[
                subject_mask
            ]

            subject_values = subject_values[
                np.isfinite(subject_values)
            ]

            if subject_values.size >= 32:

                subject_values = np.clip(
                    subject_values,
                    0.0,
                    100.0,
                )

                subject_values = sample_array(
                    subject_values,
                    50_000,
                )

                subject_median = percentile(
                    subject_values,
                    50,
                )

                subject_p25 = percentile(
                    subject_values,
                    25,
                )

                subject_p75 = percentile(
                    subject_values,
                    75,
                )

                subject_valid = True

    # ------------------------------------------------------------------
    # Semantic regions
    # ------------------------------------------------------------------

    background_mask = None
    person_mask = None

    if semantic:

        background_mask = normalized_mask(
            semantic.get("background")
        )

        person_mask = normalized_mask(
            semantic.get("person")
        )

    background_median = None
    person_median = None

    if (
        background_mask is not None
        and background_mask.shape == l.shape
    ):
        bg_values = l[
            background_mask
        ]

        bg_values = bg_values[
            np.isfinite(bg_values)
        ]

        if bg_values.size >= 32:
            background_median = percentile(
                bg_values,
                50,
            )

    if (
        person_mask is not None
        and person_mask.shape == l.shape
    ):
        person_values = l[
            person_mask
        ]

        person_values = person_values[
            np.isfinite(person_values)
        ]

        if person_values.size >= 32:
            person_median = percentile(
                person_values,
                50,
            )

    # ------------------------------------------------------------------
    # Low-key score
    # ------------------------------------------------------------------

    low_key_score = 0.0

    # Median is useful, but not sufficient.
    if p50 < 38:
        low_key_score += 0.25

    if p50 < 32:
        low_key_score += 0.15

    if shadow_ratio > 0.30:
        low_key_score += 0.20

    if shadow_ratio > 0.45:
        low_key_score += 0.10

    if highlight_ratio < 0.30:
        low_key_score += 0.10

    if bright_highlight_ratio < 0.10:
        low_key_score += 0.05

    if p90 < 75:
        low_key_score += 0.10

    if p95 < 82:
        low_key_score += 0.05

    low_key_score = clamp(
        low_key_score,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # High-key score
    # ------------------------------------------------------------------

    high_key_score = 0.0

    if p50 > 62:
        high_key_score += 0.25

    if p50 > 68:
        high_key_score += 0.15

    if highlight_ratio > 0.30:
        high_key_score += 0.20

    if highlight_ratio > 0.45:
        high_key_score += 0.10

    if shadow_ratio < 0.30:
        high_key_score += 0.10

    if deep_shadow_ratio < 0.10:
        high_key_score += 0.05

    if p10 > 25:
        high_key_score += 0.10

    if p05 > 18:
        high_key_score += 0.05

    high_key_score = clamp(
        high_key_score,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Backlit detection
    # ------------------------------------------------------------------

    backlit_score = 0.0

    if subject_valid:

        # Subject is substantially darker than the overall scene.
        if subject_median < p50 - 10:
            backlit_score += 0.30

        if subject_median < p50 - 18:
            backlit_score += 0.25

        if p90 > 80:
            backlit_score += 0.15

        if highlight_ratio > 0.20:
            backlit_score += 0.15

        if white_clip_ratio > 0.01:
            backlit_score += 0.10

    elif person_median is not None:

        if person_median < p50 - 10:
            backlit_score += 0.35

        if person_median < p50 - 18:
            backlit_score += 0.25

        if p90 > 80:
            backlit_score += 0.15

    backlit_score = clamp(
        backlit_score,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Flat / contrasty detection
    # ------------------------------------------------------------------

    contrast_score = 0.0

    if tonal_range > 70:
        contrast_score += 0.40

    if tonal_range > 82:
        contrast_score += 0.20

    if iqr > 45:
        contrast_score += 0.20

    if shadow_ratio > 0.15 and highlight_ratio > 0.15:
        contrast_score += 0.20

    contrast_score = clamp(
        contrast_score,
        0.0,
        1.0,
    )

    flat_score = 0.0

    if tonal_range < 45:
        flat_score += 0.40

    if tonal_range < 35:
        flat_score += 0.20

    if iqr < 25:
        flat_score += 0.20

    if (
        black_clip_ratio < 0.01
        and white_clip_ratio < 0.01
    ):
        flat_score += 0.10

    if abs(skew_indicator) < 0.15:
        flat_score += 0.10

    flat_score = clamp(
        flat_score,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Prevent clipping from being mistaken for high-key
    # ------------------------------------------------------------------

    # A high-key scene has many bright pixels but should not automatically
    # be interpreted as overexposed.
    #
    # If the image contains excessive white clipping, classify it separately
    # through the exposure analyzer rather than calling it simply high-key.

    severe_white_clipping = (
        white_clip_ratio > 0.08
    )

    severe_black_clipping = (
        black_clip_ratio > 0.08
    )

    # ------------------------------------------------------------------
    # Final classification
    # ------------------------------------------------------------------

    scores = {
        "low_key": low_key_score,
        "high_key": high_key_score,
        "backlit": backlit_score,
        "contrasty": contrast_score,
        "flat": flat_score,
    }

    # Backlit has priority when semantic evidence strongly supports it.
    if backlit_score >= 0.70:
        scene_type = "backlit"
        confidence = backlit_score

    elif (
        low_key_score >= 0.65
        and high_key_score < 0.45
    ):
        scene_type = "low_key"
        confidence = low_key_score

    elif (
        high_key_score >= 0.65
        and low_key_score < 0.45
    ):
        scene_type = "high_key"
        confidence = high_key_score

    elif contrast_score >= 0.72:
        scene_type = "contrasty"
        confidence = contrast_score

    elif flat_score >= 0.72:
        scene_type = "flat"
        confidence = flat_score

    else:
        scene_type = "normal"

        # Confidence for normal means "no strong special scene detected".
        strongest_special = max(
            scores.values()
        )

        confidence = clamp(
            1.0 - strongest_special,
            0.35,
            0.90,
        )

    # ------------------------------------------------------------------
    # Human-readable diagnostic flags
    # ------------------------------------------------------------------

    flags = []

    if severe_white_clipping:
        flags.append(
            "severe_highlight_clipping"
        )

    if severe_black_clipping:
        flags.append(
            "severe_shadow_clipping"
        )

    if near_white_ratio > 0.15:
        flags.append(
            "large_bright_region"
        )

    if near_black_ratio > 0.15:
        flags.append(
            "large_dark_region"
        )

    if subject_valid:
        if subject_median < 25:
            flags.append(
                "dark_subject"
            )
        elif subject_median > 78:
            flags.append(
                "bright_subject"
            )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------

    return {
        "type": scene_type,
        "confidence": round(
            float(confidence),
            4,
        ),

        "scores": {
            key: round(
                float(value),
                4,
            )
            for key, value in scores.items()
        },

        "statistics": {
            "p01_l_star": round(p01, 3),
            "p05_l_star": round(p05, 3),
            "p10_l_star": round(p10, 3),
            "p25_l_star": round(p25, 3),
            "median_l_star": round(p50, 3),
            "p75_l_star": round(p75, 3),
            "p90_l_star": round(p90, 3),
            "p95_l_star": round(p95, 3),
            "p99_l_star": round(p99, 3),

            "tonal_range": round(
                tonal_range,
                3,
            ),

            "interquartile_range": round(
                iqr,
                3,
            ),

            "shadow_ratio": round(
                shadow_ratio,
                5,
            ),

            "deep_shadow_ratio": round(
                deep_shadow_ratio,
                5,
            ),

            "highlight_ratio": round(
                highlight_ratio,
                5,
            ),

            "bright_highlight_ratio": round(
                bright_highlight_ratio,
                5,
            ),

            "black_clip_ratio": round(
                black_clip_ratio,
                5,
            ),

            "white_clip_ratio": round(
                white_clip_ratio,
                5,
            ),

            "skew_indicator": round(
                skew_indicator,
                4,
            ),
        },

        "subject": {
            "available": subject_valid,
            "median_l_star": (
                round(
                    subject_median,
                    3,
                )
                if subject_median is not None
                else None
            ),
            "p25_l_star": (
                round(
                    subject_p25,
                    3,
                )
                if subject_p25 is not None
                else None
            ),
            "p75_l_star": (
                round(
                    subject_p75,
                    3,
                )
                if subject_p75 is not None
                else None
            ),
        },

        "regions": {
            "background_median_l_star": (
                round(
                    background_median,
                    3,
                )
                if background_median is not None
                else None
            ),

            "person_median_l_star": (
                round(
                    person_median,
                    3,
                )
                if person_median is not None
                else None
            ),
        },

        "flags": flags,
    }