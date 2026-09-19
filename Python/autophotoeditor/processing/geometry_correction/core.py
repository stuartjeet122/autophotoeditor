from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from autophotoeditor.core.image_io import decode_color


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_ANALYSIS_SIZE = 1600
DEFAULT_MAX_LINES = 300

MIN_LINE_ANGLE = 18.0
MAX_LINE_ANGLE = 72.0

VP_MIN_LINES = 4
VP_MAX_LINES = 45

# Confidence thresholds.
AUTO_VP_CONFIDENCE = 0.40
AUTO_MIN_STRENGTH = 1.0

# Maximum automatic correction.
AUTO_MAX_PERSPECTIVE = 75.0
AUTO_MAX_ROTATION = 15.0

# Safety limits.
MAX_OUTPUT_MULTIPLIER = 3.0
MIN_OUTPUT_SIZE = 16

EPS = 1e-10


# ============================================================
# DATA
# ============================================================

@dataclass
class DetectedLine:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle: float
    midpoint_x: float
    midpoint_y: float


@dataclass
class VanishingPointResult:
    point: Optional[np.ndarray] = None
    confidence: float = 0.0
    inlier_ratio: float = 0.0
    support: float = 0.0
    spread: float = float("inf")


@dataclass
class GeometryEstimate:
    level: float = 0.0

    vertical_vp: Optional[np.ndarray] = None
    horizontal_vp: Optional[np.ndarray] = None

    vertical_strength: float = 0.0
    horizontal_strength: float = 0.0

    vertical_confidence: float = 0.0
    horizontal_confidence: float = 0.0

    lines: List[DetectedLine] = field(default_factory=list)


