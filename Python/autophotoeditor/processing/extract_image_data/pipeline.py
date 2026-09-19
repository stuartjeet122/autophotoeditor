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
from .scene import *
from .tone import *
from .white_balance import *
from .color import *
from .report import *


def analyze_image(
    image: np.ndarray,
    use_ai: bool = False,
) -> Dict[str, Any]:

    if image is None:
        raise ValueError(
            "Image is None."
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
            "Expected BGR image with 3 channels."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected BGR image with 3 channels."
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Analysis resize.
    # --------------------------------------------------------

    scale = min(
        1.0,
        MAX_ANALYSIS_DIM
        / max(
            original_height,
            original_width,
        ),
    )

    if scale < 1.0:

        analysis_width = max(
            1,
            int(
                original_width
                * scale
            ),
        )

        analysis_height = max(
            1,
            int(
                original_height
                * scale
            ),
        )

        analysis_image = cv2.resize(
            image,
            (
                analysis_width,
                analysis_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:
        analysis_image = image

    h, w = analysis_image.shape[:2]

    # --------------------------------------------------------
    # This engine reads the image directly and computes enhancement values for
    # luminosity and neutral tone. It does not use mask JSON or mask-based
    # regions in the main analysis flow.
    # --------------------------------------------------------

    subject = None
    subject_source = "none"
    semantic: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------
    # Core color spaces.
    # --------------------------------------------------------

    rgb = bgr_to_rgb_float(
        analysis_image
    )

    rgb_linear = srgb_to_linear(
        rgb
    )

    y_linear = rgb_to_linear_luminance(
        rgb_linear
    )

    lab = bgr_to_lab(
        analysis_image
    )

    l_star = lab_l_star(
        lab
    )

    # --------------------------------------------------------
    # Scene.
    # --------------------------------------------------------

    classification = classify_scene(
        l_star,
        semantic,
        subject,
    )

    scene_type = classification[
        "type"
    ]

    # --------------------------------------------------------
    # Tone.
    # --------------------------------------------------------

    tone = analyze_tone(
        y_linear,
        l_star,
    )

    # --------------------------------------------------------
    # Exposure optimizer.
    # --------------------------------------------------------

    exposure = derive_exposure_recommendation(
        y_linear,
        subject,
        scene_type,
    )

    exposure_ev = safe_float(
        exposure.get(
            "value"
        ),
        0.0,
    )

    # --------------------------------------------------------
    # Lightroom-style light controls.
    # --------------------------------------------------------

    tone_controls = derive_lightroom_tone_controls(
        y_linear,
        exposure_ev,
        scene_type,
        subject,
    )

    # --------------------------------------------------------
    # White balance.
    # --------------------------------------------------------

    wb = analyze_white_balance(
        analysis_image,
        semantic,
        subject,
    )

    # --------------------------------------------------------
    # Color.
    # --------------------------------------------------------

    color = analyze_color(
        analysis_image,
        semantic,
    )

    # --------------------------------------------------------
    # Detail.
    # --------------------------------------------------------

    detail = analyze_detail(
        analysis_image
    )

    # --------------------------------------------------------
    # Lighting quality.
    # --------------------------------------------------------

    lighting = analyze_lighting_quality(
        y_linear,
        scene_type,
        subject,
    )

    # --------------------------------------------------------
    # Region analysis is intentionally disabled here. This engine only scans the
    # image and emits luminosity and neutral-tone enhancement values.
    # --------------------------------------------------------

    regions: list[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Grid.
    # --------------------------------------------------------

    grid = analyze_grid(
        y_linear
    )

    # --------------------------------------------------------
    # Histograms.
    # --------------------------------------------------------

    histogram_luminance = calculate_histogram(
        np.clip(
            y_linear,
            0.0,
            1.0,
        )
    )

    histogram_l_star = calculate_histogram(
        np.clip(
            l_star / 100.0,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Quality.
    # --------------------------------------------------------

    quality = analyze_quality(
        analysis_image,
        tone,
        detail,
    )

    # --------------------------------------------------------
    # Compatibility recommendations.
    # --------------------------------------------------------

    recommendations = derive_edit_recommendations(
        exposure,
        tone_controls,
        wb,
        color,
        detail,
    )

    # --------------------------------------------------------
    # New correction plan.
    # --------------------------------------------------------

    correction_plan = derive_correction_plan(
        exposure,
        tone_controls,
        wb,
        color,
    )

    # --------------------------------------------------------
    # Neutral tone summary.
    # --------------------------------------------------------

    neutral_balance = {
        "technical": {
            "gains_rgb": wb.get(
                "gains_rgb",
                [1.0, 1.0, 1.0],
            ),
            "temperature_cast": wb.get(
                "temperature_cast",
                0.0,
            ),
            "tint_cast": wb.get(
                "tint_cast",
                0.0,
            ),
        },
        "recommended": {
            "temperature": (
                wb.get(
                    "temperature_cast",
                    0.0,
                )
            ),
            "tint": (
                wb.get(
                    "tint_cast",
                    0.0,
                )
            ),
        },
        "neutrality_before": wb.get(
            "neutrality_before",
            100.0,
        ),
        "neutrality_after": wb.get(
            "neutrality_after",
            100.0,
        ),
        "neutrality_improvement_percentage": wb.get(
            "neutrality_improvement_percentage",
            0.0,
        ),
        "confidence": wb.get(
            "confidence",
            0.0,
        ),
        "method": wb.get(
            "method",
            "none",
        ),
    }

    # --------------------------------------------------------
    # Auto tone summary.
    # --------------------------------------------------------

    auto_tone = {
        "exposure": {
            "value": exposure.get(
                "value",
                0.0,
            ),
            "confidence": exposure.get(
                "confidence",
                0.0,
            ),
        },
        "contrast": tone_controls[
            "contrast"
        ],
        "highlights": tone_controls[
            "highlights"
        ],
        "shadows": tone_controls[
            "shadows"
        ],
        "whites": tone_controls[
            "whites"
        ],
        "blacks": tone_controls[
            "blacks"
        ],
        "vibrance": color[
            "vibrance"
        ],
        "saturation": color[
            "saturation"
        ],
        "tone_confidence": exposure.get(
            "confidence",
            0.0,
        ),
    }

    # --------------------------------------------------------
    # Return schema.
    # --------------------------------------------------------

    result: Dict[str, Any] = {

        "schema_version": SCHEMA_VERSION,

        "image": {
            "width": original_width,
            "height": original_height,
            "analysis_width": w,
            "analysis_height": h,
            "scale": round(
                scale,
                6,
            ),
        },

        "classification": classification,

        "scene": classification,

        "lighting_quality": lighting,

        "tone": tone,

        "tone_distribution": {
            "linear_luminance": tone[
                "linear_luminance"
            ],
            "l_star": tone[
                "l_star"
            ],
        },

        "exposure_and_tone": {
            "exposure": exposure,
            "controls": tone_controls,
        },

        "white_balance": wb,

        "neutral_balance": neutral_balance,

        "color": color,

        "detail": detail,

        "quality": quality,

        "regions": regions,

        "grid": grid,

        "histograms": {
            "luminance": histogram_luminance,
            "l_star": histogram_l_star,
        },

        "auto_tone": auto_tone,

        "recommendations": recommendations,

        "correction_plan": correction_plan,

        "white_balance_diagnostics": {
            "neutral_patch_count": wb.get(
                "neutral_patch_count",
                0,
            ),
            "neutral_coverage": wb.get(
                "neutral_coverage",
                0.0,
            ),
            "estimators": wb.get(
                "estimators",
                [],
            ),
        },

        "ai": {
            "enabled": bool(use_ai),
            "used": False,
        },
    }

    return result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def analyze(
    image: np.ndarray,
    use_ai: bool = False,
) -> Dict[str, Any]:

    return analyze_image(
        image=image,
        use_ai=use_ai,
    )


def process(
    image: np.ndarray,
    use_ai: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:

    """Compatibility wrapper for direct image-only enhancement analysis."""

    return analyze_image(
        image=image,
        use_ai=use_ai,
    )


# ============================================================
# IMAGE INPUT
# ============================================================

def load_image(
    path: Path,
) -> np.ndarray:

    image = decode_color(
        path
    )

    if image is None:
        raise RuntimeError(
            f"Unable to decode image: {path}"
        )

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(
            f"Unsupported image shape: {image.shape}"
        )

    return image


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "AutoPhotoEditor Image Analyzer V4"
        )
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )

    parser.add_argument(
        "--ai",
        action="store_true",
        help=(
            "Reserved for optional AI analysis. "
            "No AI model is loaded by default."
        ),
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.debug
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        logger.error(
            "Input image does not exist: %s",
            input_path,
        )
        return 1

    try:

        image = load_image(
            input_path
        )

        result = analyze_image(
            image=image,
            use_ai=args.ai,
        )

        if args.pretty:

            output_text = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )

        else:

            output_text = json.dumps(
                result,
                separators=(
                    ",",
                    ":",
                ),
                ensure_ascii=False,
            )

        if args.output:

            output_path = Path(
                args.output
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                output_text,
                encoding="utf-8",
            )

            logger.info(
                "Analysis written to %s",
                output_path,
            )

        else:

            print(
                output_text
            )

        return 0

    except Exception as exc:

        logger.exception(
            "Image analysis failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
