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


# ============================================================
# IMAGE IO
# ============================================================

def read_image(path: str) -> np.ndarray:
    """
    Decode an image using the project's image IO layer.
    """
    image = decode_color(Path(path))

    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")

    if not isinstance(image, np.ndarray):
        raise RuntimeError("Image decoder returned an invalid object.")

    if image.ndim != 3:
        raise RuntimeError("Input image must be a color image.")

    if image.shape[2] != 3:
        raise RuntimeError(
            f"Expected 3-channel BGR image, got {image.shape[2]} channels."
        )

    return image


def write_image(
    path: str,
    image: np.ndarray,
    quality: int = 95,
) -> None:
    """
    Write an image with sensible codec settings.
    """
    output = Path(path)

    parent = output.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)

    if image is None or image.size == 0:
        raise RuntimeError("Cannot write an empty image.")

    ext = output.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(np.clip(quality, 1, 100)),
        ]

    elif ext == ".png":
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    elif ext == ".webp":
        params = [
            cv2.IMWRITE_WEBP_QUALITY,
            int(np.clip(quality, 1, 100)),
        ]

    elif ext in (".bmp", ".tif", ".tiff"):
        params = []

    else:
        raise RuntimeError(
            f"Unsupported output format: {ext or '<none>'}"
        )

    ok = cv2.imwrite(
        str(output),
        image,
        params,
    )

    if not ok or not output.exists():
        raise RuntimeError(
            f"Could not write output image: {output}"
        )


# ============================================================
# ANGLES
# ============================================================

def normalize_line_angle(angle: float) -> float:
    """
    Normalize an undirected line orientation to [-90, 90).
    """
    return ((float(angle) + 90.0) % 180.0) - 90.0


def angle_difference(a: float, b: float) -> float:
    """
    Smallest orientation difference between two undirected lines.
    """
    return abs(
        normalize_line_angle(
            normalize_line_angle(a) - normalize_line_angle(b)
        )
    )


def circular_line_mean(
    angles: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted mean for undirected line orientations.

    Doubling the angle converts the 180-degree periodicity
    into a normal 360-degree circular mean.
    """
    if len(angles) == 0:
        return 0.0

    radians = np.deg2rad(angles * 2.0)

    sx = float(np.sum(np.cos(radians) * weights))
    sy = float(np.sum(np.sin(radians) * weights))

    if abs(sx) < EPS and abs(sy) < EPS:
        return float(angles[0])

    result = math.degrees(
        0.5 * math.atan2(sy, sx)
    )

    return normalize_line_angle(result)


# ============================================================
# LINE DETECTION
# ============================================================

def detect_lines(
    image: np.ndarray,
    analysis_size: int = DEFAULT_ANALYSIS_SIZE,
    max_lines: int = DEFAULT_MAX_LINES,
) -> List[DetectedLine]:
    """
    Detect structural lines using Canny + probabilistic Hough.

    Coordinates are always returned in original-image pixels.
    """
    h, w = image.shape[:2]

    if h < 32 or w < 32:
        return []

    analysis_size = max(256, int(analysis_size))
    max_lines = max(20, int(max_lines))

    scale = min(
        1.0,
        analysis_size / float(max(h, w)),
    )

    if scale < 1.0:
        small = cv2.resize(
            image,
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    # Mild smoothing suppresses pixel noise while preserving
    # architectural edges.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Robust automatic Canny thresholds.
    median = float(np.median(gray))

    low = int(
        np.clip(
            median * 0.60,
            20,
            100,
        )
    )

    high = int(
        np.clip(
            median * 1.40,
            70,
            240,
        )
    )

    if high <= low:
        high = min(
            255,
            low + 50,
        )

    edges = cv2.Canny(
        gray,
        low,
        high,
        apertureSize=3,
        L2gradient=True,
    )

    sh, sw = gray.shape

    diagonal = math.hypot(
        sw,
        sh,
    )

    min_length_small = max(
        30.0,
        diagonal * 0.035,
    )

    max_gap_small = max(
        5,
        int(diagonal * 0.015),
    )

    threshold = max(
        20,
        int(diagonal * 0.020),
    )

    raw = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 360.0,
        threshold=threshold,
        minLineLength=int(min_length_small),
        maxLineGap=max_gap_small,
    )

    if raw is None:
        return []

    raw = np.asarray(raw).reshape(-1, 4)

    min_length_full = min_length_small / max(scale, EPS)

    lines: List[DetectedLine] = []

    for x1, y1, x2, y2 in raw:
        x1 = float(x1) / scale
        y1 = float(y1) / scale
        x2 = float(x2) / scale
        y2 = float(y2) / scale

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(
            dx,
            dy,
        )

        if not math.isfinite(length):
            continue

        if length < min_length_full:
            continue

        angle = normalize_line_angle(
            math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )
        )

        midpoint_x = (x1 + x2) * 0.5
        midpoint_y = (y1 + y2) * 0.5

        lines.append(
            DetectedLine(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length=length,
                angle=angle,
                midpoint_x=midpoint_x,
                midpoint_y=midpoint_y,
            )
        )

    if not lines:
        return []

    # Longest first.
    lines.sort(
        key=lambda line: line.length,
        reverse=True,
    )

    # --------------------------------------------------------
    # Deduplicate nearly identical Hough segments.
    #
    # This is deliberately conservative: two parallel lines
    # separated by a meaningful distance are retained.
    # --------------------------------------------------------

    kept: List[DetectedLine] = []

    diag_full = math.hypot(
        w,
        h,
    )

    midpoint_threshold = diag_full * 0.012

    for line in lines:
        duplicate = False

        for old in kept:
            if angle_difference(
                line.angle,
                old.angle,
            ) > 2.5:
                continue

            distance = math.hypot(
                line.midpoint_x - old.midpoint_x,
                line.midpoint_y - old.midpoint_y,
            )

            if distance <= midpoint_threshold:
                if line.length <= old.length * 1.30:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(line)

        if len(kept) >= max_lines:
            break

    return kept


# ============================================================
# ORIENTATION GROUPS
# ============================================================

def split_orientation(
    lines: List[DetectedLine],
) -> Tuple[
    List[DetectedLine],
    List[DetectedLine],
]:
    """
    Split lines into approximately horizontal and vertical groups.

    Lines outside these ranges are deliberately ignored for the
    basic architectural VP estimator.
    """
    horizontal: List[DetectedLine] = []
    vertical: List[DetectedLine] = []

    for line in lines:
        a = abs(line.angle)

        if a <= MIN_LINE_ANGLE:
            horizontal.append(line)

        elif a >= MAX_LINE_ANGLE:
            vertical.append(line)

    return horizontal, vertical


# ============================================================
# ROBUST LEVEL ESTIMATION
# ============================================================

def weighted_orientation(
    lines: List[DetectedLine],
) -> Optional[float]:
    """
    Robust orientation estimate using a circular mean after
    removing obvious angular outliers.
    """
    if not lines:
        return None

    angles = np.asarray(
        [line.angle for line in lines],
        dtype=np.float64,
    )

    weights = np.asarray(
        [
            math.sqrt(max(1.0, line.length))
            for line in lines
        ],
        dtype=np.float64,
    )

    if len(angles) == 1:
        return float(angles[0])

    initial = circular_line_mean(
        angles,
        weights,
    )

    residuals = np.asarray(
        [
            angle_difference(
                float(a),
                initial,
            )
            for a in angles
        ],
        dtype=np.float64,
    )

    # Keep lines close to the dominant direction.
    threshold = max(
        3.0,
        float(np.percentile(residuals, 70)),
    )

    mask = residuals <= threshold

    if np.sum(mask) < 1:
        return initial

    return circular_line_mean(
        angles[mask],
        weights[mask],
    )


def estimate_level(
    horizontal: List[DetectedLine],
) -> float:
    """
    Estimate image roll from approximately horizontal lines.

    Perspective causes horizontal architectural lines to vary,
    so only a conservative correction is applied automatically.
    """
    if not horizontal:
        return 0.0

    angle = weighted_orientation(
        horizontal
    )

    if angle is None:
        return 0.0

    return float(
        np.clip(
            angle,
            -AUTO_MAX_ROTATION,
            AUTO_MAX_ROTATION,
        )
    )


# ============================================================
# HOMOGENEOUS LINE MATH
# ============================================================

def line_from_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Optional[np.ndarray]:
    """
    Return normalized homogeneous line ax + by + c = 0.
    """
    p1 = np.array(
        [x1, y1, 1.0],
        dtype=np.float64,
    )

    p2 = np.array(
        [x2, y2, 1.0],
        dtype=np.float64,
    )

    line = np.cross(
        p1,
        p2,
    )

    norm = math.hypot(
        float(line[0]),
        float(line[1]),
    )

    if norm < EPS:
        return None

    return line / norm


def intersection(
    l1: np.ndarray,
    l2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Intersect two homogeneous lines.
    """
    p = np.cross(
        l1,
        l2,
    )

    denominator = float(p[2])

    if abs(denominator) < EPS:
        return None

    result = p[:2] / denominator

    if not np.all(np.isfinite(result)):
        return None

    return result


# ============================================================
# VANISHING POINT ESTIMATION
# ============================================================

def _line_angle_error_to_point(
    line: DetectedLine,
    point: np.ndarray,
) -> float:
    """
    Angular disagreement between a detected segment and the
    direction from its midpoint toward a candidate VP.
    """
    dx = float(point[0]) - line.midpoint_x
    dy = float(point[1]) - line.midpoint_y

    if math.hypot(dx, dy) < EPS:
        return 90.0

    target_angle = normalize_line_angle(
        math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )
    )

    return angle_difference(
        line.angle,
        target_angle,
    )


def robust_vanishing_point(
    lines: List[DetectedLine],
    width: int,
    height: int,
    axis: str,
) -> VanishingPointResult:
    """
    Robust VP estimator.

    Instead of taking a simple median of pairwise intersections,
    candidate VPs are scored by how many actual detected lines
    point toward them.

    This substantially reduces false VPs generated by unrelated
    objects.
    """
    if len(lines) < VP_MIN_LINES:
        return VanishingPointResult()

    axis = axis.lower()

    if axis not in (
        "vertical",
        "horizontal",
    ):
        raise ValueError(
            f"Unknown VP axis: {axis}"
        )

    strongest = sorted(
        lines,
        key=lambda line: line.length,
        reverse=True,
    )[:VP_MAX_LINES]

    hom_lines = []

    for line in strongest:
        hline = line_from_points(
            line.x1,
            line.y1,
            line.x2,
            line.y2,
        )

        if hline is not None:
            hom_lines.append(
                (
                    line,
                    hline,
                )
            )

    if len(hom_lines) < VP_MIN_LINES:
        return VanishingPointResult()

    diag = math.hypot(
        width,
        height,
    )

    # VPs farther than this are possible, but beyond this point
    # the estimate becomes increasingly numerically weak.
    coordinate_limit = diag * 20.0

    candidates: List[
        Tuple[np.ndarray, float]
    ] = []

    for i in range(len(hom_lines)):
        line1, h1 = hom_lines[i]

        for j in range(i + 1, len(hom_lines)):
            line2, h2 = hom_lines[j]

            # Reject nearly parallel lines.
            cross_norm = abs(
                float(
                    h1[0] * h2[1]
                    - h1[1] * h2[0]
                )
            )

            if cross_norm < 0.015:
                continue

            point = intersection(
                h1,
                h2,
            )

            if point is None:
                continue

            x = float(point[0])
            y = float(point[1])

            if (
                abs(x) > coordinate_limit
                or abs(y) > coordinate_limit
            ):
                continue

            pair_weight = math.sqrt(
                max(
                    1.0,
                    line1.length * line2.length,
                )
            )

            candidates.append(
                (
                    point,
                    pair_weight,
                )
            )

    if len(candidates) < 3:
        return VanishingPointResult()

    # --------------------------------------------------------
    # Score every candidate against all strong lines.
    # --------------------------------------------------------

    best_point: Optional[np.ndarray] = None
    best_score = -float("inf")
    best_inliers = 0
    best_support = 0.0

    angle_tolerance = 5.0

    for candidate, candidate_weight in candidates:
        total_weight = 0.0
        inlier_weight = 0.0
        inliers = 0

        for line, _ in hom_lines:
            error = _line_angle_error_to_point(
                line,
                candidate,
            )

            line_weight = math.sqrt(
                max(
                    1.0,
                    line.length,
                )
            )

            total_weight += line_weight

            if error <= angle_tolerance:
                # Smooth angular contribution.
                angular_score = max(
                    0.0,
                    1.0 - error / angle_tolerance,
                )

                inlier_weight += (
                    line_weight
                    * angular_score
                )

                inliers += 1

        if total_weight <= EPS:
            continue

        ratio = inlier_weight / total_weight

        score = (
            ratio
            * math.log1p(candidate_weight)
            * (1.0 + 0.20 * inliers)
        )

        if score > best_score:
            best_score = score
            best_point = candidate.copy()
            best_inliers = inliers
            best_support = ratio

    if best_point is None:
        return VanishingPointResult()

    # --------------------------------------------------------
    # Refine VP using intersections from lines that actually
    # agree with the candidate.
    # --------------------------------------------------------

    refined_points = []
    refined_weights = []

    for candidate, candidate_weight in candidates:
        distance = float(
            np.linalg.norm(
                candidate - best_point
            )
        )

        if distance > diag * 0.35:
            continue

        refined_points.append(candidate)
        refined_weights.append(
            candidate_weight
        )

    if len(refined_points) >= 2:
        pts = np.asarray(
            refined_points,
            dtype=np.float64,
        )

        weights = np.asarray(
            refined_weights,
            dtype=np.float64,
        )

        weights = np.maximum(
            weights,
            1.0,
        )

        refined = np.average(
            pts,
            axis=0,
            weights=weights,
        )

        if np.all(np.isfinite(refined)):
            best_point = refined

    # --------------------------------------------------------
    # Final confidence.
    # --------------------------------------------------------

    errors = []

    for line, _ in hom_lines:
        errors.append(
            _line_angle_error_to_point(
                line,
                best_point,
            )
        )

    errors_array = np.asarray(
        errors,
        dtype=np.float64,
    )

    inlier_mask = errors_array <= angle_tolerance

    if not np.any(inlier_mask):
        return VanishingPointResult()

    inlier_ratio = float(
        np.mean(inlier_mask)
    )

    weighted_errors = errors_array[
        inlier_mask
    ]

    spread = float(
        np.median(weighted_errors)
    )

    # Confidence combines support, number of lines,
    # and angular consistency.
    consistency = float(
        np.clip(
            1.0 - spread / angle_tolerance,
            0.0,
            1.0,
        )
    )

    line_count_factor = float(
        np.clip(
            len(hom_lines) / 12.0,
            0.0,
            1.0,
        )
    )

    confidence = (
        0.55 * best_support
        + 0.25 * inlier_ratio
        + 0.20 * consistency
    )

    confidence *= (
        0.65
        + 0.35 * line_count_factor
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    return VanishingPointResult(
        point=best_point,
        confidence=confidence,
        inlier_ratio=inlier_ratio,
        support=best_support,
        spread=spread,
    )


# ============================================================
# PERSPECTIVE STRENGTH
# ============================================================

def _vp_offset_normalized(
    vp: np.ndarray,
    width: int,
    height: int,
    axis: str,
) -> float:
    """
    Signed VP displacement.

    Vertical perspective is determined primarily by horizontal
    VP displacement.

    Horizontal perspective is determined primarily by vertical
    VP displacement.
    """
    cx = width * 0.5
    cy = height * 0.5

    if axis == "vertical":
        return float(
            (float(vp[0]) - cx)
            / max(
                1.0,
                width,
            )
        )

    return float(
        (float(vp[1]) - cy)
        / max(
            1.0,
            height,
        )
    )


def perspective_amount(
    vp: Optional[np.ndarray],
    width: int,
    height: int,
    confidence: float,
    axis: str,
) -> float:
    """
    Convert VP displacement into a signed correction amount.

    IMPORTANT:
    The sign is retained.

    A VP to the left of center and a VP to the right of center
    require opposite projective corrections.
    """
    if vp is None:
        return 0.0

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    normalized = _vp_offset_normalized(
        vp,
        width,
        height,
        axis,
    )

    # Scale so typical architectural perspective lands in a
    # useful slider range without making weak detections huge.
    amount = (
        normalized
        * 180.0
        * confidence
    )

    return float(
        np.clip(
            amount,
            -100.0,
            100.0,
        )
    )


# ============================================================
# GEOMETRY ANALYSIS
# ============================================================

def analyze_geometry(
    image: np.ndarray,
    max_lines: int = DEFAULT_MAX_LINES,
    verbose: bool = True,
) -> GeometryEstimate:
    """
    Analyze structural image geometry.
    """
    h, w = image.shape[:2]

    if verbose:
        print(
            "[1/4] Analysing image geometry...",
            flush=True,
        )

    lines = detect_lines(
        image,
        analysis_size=DEFAULT_ANALYSIS_SIZE,
        max_lines=max_lines,
    )

    horizontal, vertical = split_orientation(
        lines
    )

    if verbose:
        print(
            f"      Structural lines: {len(lines)}",
            flush=True,
        )

        print(
            f"      Horizontal candidates: "
            f"{len(horizontal)}",
            flush=True,
        )

        print(
            f"      Vertical candidates: "
            f"{len(vertical)}",
            flush=True,
        )

    level = estimate_level(
        horizontal
    )

    if verbose:
        print(
            f"      Estimated level: "
            f"{level:.3f}°",
            flush=True,
        )

    if verbose:
        print(
            "[2/4] Estimating perspective...",
            flush=True,
        )

    vertical_result = robust_vanishing_point(
        vertical,
        w,
        h,
        "vertical",
    )

    horizontal_result = robust_vanishing_point(
        horizontal,
        w,
        h,
        "horizontal",
    )

    vertical_amount = perspective_amount(
        vertical_result.point,
        w,
        h,
        vertical_result.confidence,
        "vertical",
    )

    horizontal_amount = perspective_amount(
        horizontal_result.point,
        w,
        h,
        horizontal_result.confidence,
        "horizontal",
    )

    if verbose:
        if vertical_result.point is not None:
            print(
                f"      Vertical VP: "
                f"({vertical_result.point[0]:.1f}, "
                f"{vertical_result.point[1]:.1f})",
                flush=True,
            )

            print(
                f"      Vertical confidence: "
                f"{vertical_result.confidence * 100:.1f}%",
                flush=True,
            )

            print(
                f"      Vertical correction: "
                f"{vertical_amount:.2f}",
                flush=True,
            )
        else:
            print(
                "      Vertical VP: none",
                flush=True,
            )

        if horizontal_result.point is not None:
            print(
                f"      Horizontal VP: "
                f"({horizontal_result.point[0]:.1f}, "
                f"{horizontal_result.point[1]:.1f})",
                flush=True,
            )

            print(
                f"      Horizontal confidence: "
                f"{horizontal_result.confidence * 100:.1f}%",
                flush=True,
            )

            print(
                f"      Horizontal correction: "
                f"{horizontal_amount:.2f}",
                flush=True,
            )
        else:
            print(
                "      Horizontal VP: none",
                flush=True,
            )

    return GeometryEstimate(
        level=level,
        vertical_vp=vertical_result.point,
        horizontal_vp=horizontal_result.point,
        vertical_strength=vertical_amount,
        horizontal_strength=horizontal_amount,
        vertical_confidence=vertical_result.confidence,
        horizontal_confidence=horizontal_result.confidence,
        lines=lines,
    )


# ============================================================
# HOMOGRAPHIES
# ============================================================

def translation(
    tx: float,
    ty: float,
) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, float(tx)],
            [0.0, 1.0, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def centered_matrix(
    matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Apply a matrix around image center.
    """
    cx = width * 0.5
    cy = height * 0.5

    return (
        translation(cx, cy)
        @ matrix
        @ translation(-cx, -cy)
    )


# ============================================================
# PROJECTIVE PERSPECTIVE CORRECTION
# ============================================================

def _single_axis_projective(
    signed_strength: float,
    width: int,
    height: int,
    axis: str,
) -> np.ndarray:
    """
    Construct a projective correction from signed VP displacement.

    The denominator is centered on the image. This is important:
    using raw image coordinates makes the result dependent on
    resolution and causes excessive correction.

    For vertical correction:
        denominator = 1 + p * (x - cx)

    For horizontal correction:
        denominator = 1 + p * (y - cy)

    The sign is derived from the VP direction.
    """
    strength = float(
        np.clip(
            signed_strength / 100.0,
            -1.0,
            1.0,
        )
    )

    if abs(strength) < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    if axis == "vertical":
        dimension = max(
            1.0,
            float(width),
        )
    else:
        dimension = max(
            1.0,
            float(height),
        )

    vp_displacement = max(
        1.0,
        abs(strength) * dimension / 1.8,
    )

    p = -math.copysign(
        1.0 / vp_displacement,
        strength,
    )

    if axis == "vertical":
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [p,   0.0, 1.0],
            ],
            dtype=np.float64,
        )

    else:
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, p,   1.0],
            ],
            dtype=np.float64,
        )

    return centered_matrix(
        matrix,
        width,
        height,
    )


def _combined_projective_from_strengths(
    vertical_strength: float,
    horizontal_strength: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Combine vertical and horizontal projective correction.

    This uses a centered denominator:

        w = 1 + px*(x-cx) + py*(y-cy)

    so the correction remains resolution-aware.
    """
    vx = float(
        np.clip(
            vertical_strength / 100.0,
            -1.0,
            1.0,
        )
    )

    hy = float(
        np.clip(
            horizontal_strength / 100.0,
            -1.0,
            1.0,
        )
    )

    if abs(vx) < 1e-8 and abs(hy) < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    px = (
        -math.copysign(
            1.8,
            vx,
        )
        / max(
            1.0,
            abs(vx) * width,
        )
        if abs(vx) >= 1e-8
        else 0.0
    )

    py = (
        -math.copysign(
            1.8,
            hy,
        )
        / max(
            1.0,
            abs(hy) * height,
        )
        if abs(hy) >= 1e-8
        else 0.0
    )

    # Work in centered coordinates.
    cx = width * 0.5
    cy = height * 0.5

    centered = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [px,  py,  1.0],
        ],
        dtype=np.float64,
    )

    return (
        translation(cx, cy)
        @ centered
        @ translation(-cx, -cy)
    )


def vp_projective_transform(
    vertical_amount: float,
    horizontal_amount: float,
    width: int,
    height: int,
    vertical_vp: Optional[np.ndarray] = None,
    horizontal_vp: Optional[np.ndarray] = None,
    vertical_reference: Optional[float] = None,
    horizontal_reference: Optional[float] = None,
) -> np.ndarray:
    """
    Safe projective perspective correction.

    Unlike the original implementation, this function preserves
    the direction of the correction.
    """
    vertical_amount = float(
        np.clip(
            vertical_amount,
            -100.0,
            100.0,
        )
    )

    horizontal_amount = float(
        np.clip(
            horizontal_amount,
            -100.0,
            100.0,
        )
    )

    cx = width * 0.5
    cy = height * 0.5
    center = np.array([cx, cy], dtype=np.float64)
    vertical_point = None if vertical_vp is None else np.asarray(vertical_vp, dtype=np.float64) - center
    horizontal_point = None if horizontal_vp is None else np.asarray(horizontal_vp, dtype=np.float64) - center

    if vertical_point is None and abs(vertical_amount) > 1e-8:
        vertical_point = np.array([width * vertical_amount / 180.0, 0.0])
    if horizontal_point is None and abs(horizontal_amount) > 1e-8:
        horizontal_point = np.array([0.0, height * horizontal_amount / 180.0])

    rectifier = np.eye(3, dtype=np.float64)

    if vertical_point is not None and abs(vertical_amount) > 1e-8:
        dx, dy = vertical_point
        if abs(dx) > 1e-8:
            rectifier = np.array(
                [[dy, -dx, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -dx]],
                dtype=np.float64,
            )

    if horizontal_point is not None and abs(horizontal_amount) > 1e-8:
        dx, dy = horizontal_point
        if abs(dy) > 1e-8:
            horizontal_rectifier = np.array(
                [[1.0, 0.0, 0.0], [dy, -dx, 0.0], [0.0, 1.0, -dy]],
                dtype=np.float64,
            )
            if vertical_point is not None and abs(vertical_amount) > 1e-8:
                vertical_rectifier = rectifier
                rectifier = np.array(
                    [
                        [vertical_point[1], -vertical_point[0], 0.0],
                        [horizontal_point[1], -horizontal_point[0], 0.0],
                        np.cross(
                            np.r_[vertical_point, 1.0],
                            np.r_[horizontal_point, 1.0],
                        ),
                    ],
                    dtype=np.float64,
                )
            else:
                rectifier = horizontal_rectifier

    if np.allclose(rectifier, np.eye(3)):
        return _combined_projective_from_strengths(
            vertical_amount,
            horizontal_amount,
            width,
            height,
        )

    if abs(rectifier[2, 2]) < 1e-8:
        return _combined_projective_from_strengths(
            vertical_amount,
            horizontal_amount,
            width,
            height,
        )

    rectifier /= rectifier[2, 2]
    reference_strengths = [
        abs(vertical_amount / vertical_reference)
        if vertical_reference is not None and abs(vertical_reference) > 1e-8
        else 0.0,
        abs(horizontal_amount / horizontal_reference)
        if horizontal_reference is not None and abs(horizontal_reference) > 1e-8
        else 0.0,
    ]
    strength = max(reference_strengths)
    if strength <= 1e-8:
        strength = max(abs(vertical_amount), abs(horizontal_amount)) / 100.0
    strength = float(np.clip(strength, 0.0, 1.0))
    blended = np.eye(3, dtype=np.float64) + strength * (rectifier - np.eye(3))

    return centered_matrix(blended, width, height)


# ============================================================
# ROTATION
# ============================================================

def rotation_homography(
    degrees: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Rotation around image center.
    """
    degrees = float(degrees)

    if abs(degrees) < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    center = (
        width * 0.5,
        height * 0.5,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        degrees,
        1.0,
    )

    result = np.eye(
        3,
        dtype=np.float64,
    )

    result[:2, :] = matrix

    return result


# ============================================================
# ASPECT
# ============================================================

def aspect_homography(
    amount: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Horizontal aspect adjustment.
    """
    amount = float(
        np.clip(
            amount,
            -100.0,
            100.0,
        )
    )

    if abs(amount) < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    if amount >= 0.0:
        sx = (
            1.0
            + 0.25 * amount / 100.0
        )
    else:
        sx = 1.0 / (
            1.0
            + 0.25 * (-amount) / 100.0
        )

    matrix = np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return centered_matrix(
        matrix,
        width,
        height,
    )


# ============================================================
# SCALE
# ============================================================

def scale_homography(
    scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Centered image scaling.
    """
    scale = float(
        np.clip(
            scale,
            5.0,
            500.0,
        )
    )

    if abs(scale - 100.0) < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    s = scale / 100.0

    matrix = np.array(
        [
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return centered_matrix(
        matrix,
        width,
        height,
    )


# ============================================================
# OFFSET
# ============================================================

def offset_homography(
    x: float,
    y: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Offset in percentage of original image dimensions.
    """
    return translation(
        width * float(x) / 100.0,
        height * float(y) / 100.0,
    )


# ============================================================
# HOMOGRAPHY VALIDATION
# ============================================================

def validate_homography(
    H: np.ndarray,
) -> None:
    """
    Basic numerical sanity check.
    """
    if H.shape != (3, 3):
        raise RuntimeError(
            "Transform must be a 3x3 matrix."
        )

    if not np.all(np.isfinite(H)):
        raise RuntimeError(
            "Transform contains non-finite values."
        )

    determinant = float(
        np.linalg.det(H)
    )

    if abs(determinant) < 1e-12:
        raise RuntimeError(
            "Transform is singular."
        )


# ============================================================
# AUTO GEOMETRY
# ============================================================

def build_transform(
    image: np.ndarray,
    mode: str = "auto",
    rotate: float = 0.0,
    vertical_strength: Optional[float] = None,
    horizontal_strength: Optional[float] = None,
    aspect: float = 0.0,
    scale: float = 100.0,
    x: float = 0.0,
    y: float = 0.0,
    max_lines: int = DEFAULT_MAX_LINES,
    verbose: bool = True,
) -> Tuple[np.ndarray, GeometryEstimate]:
    """
    Build the complete geometry transform.
    """
    h, w = image.shape[:2]

    automatic_modes = {
        "auto",
        "level",
        "vertical",
        "horizontal",
        "full",
    }

    if mode not in automatic_modes and mode != "off":
        raise ValueError(
            f"Unknown geometry mode: {mode}"
        )

    if mode in automatic_modes:
        info = analyze_geometry(
            image,
            max_lines=max_lines,
            verbose=verbose,
        )
    else:
        info = GeometryEstimate()

    auto_v = info.vertical_strength
    auto_h = info.horizontal_strength

    # --------------------------------------------------------
    # Select automatic mode.
    # --------------------------------------------------------

    if mode == "auto":
        vc = info.vertical_confidence
        hc = info.horizontal_confidence

        use_vertical = (
            info.vertical_vp is not None
            and vc >= AUTO_VP_CONFIDENCE
            and abs(auto_v) >= AUTO_MIN_STRENGTH
        )

        use_horizontal = (
            info.horizontal_vp is not None
            and hc >= AUTO_VP_CONFIDENCE
            and abs(auto_h) >= AUTO_MIN_STRENGTH
        )

        if use_vertical and use_horizontal:
            actual_mode = "full"

        elif use_vertical:
            actual_mode = "vertical"

        elif use_horizontal:
            actual_mode = "horizontal"

        else:
            actual_mode = "level"

        if verbose:
            print(
                f"[3/4] Auto mode selected: "
                f"{actual_mode}",
                flush=True,
            )

    else:
        actual_mode = mode

        if verbose:
            print(
                f"[3/4] Applying mode: "
                f"{actual_mode}",
                flush=True,
            )

    # --------------------------------------------------------
    # Determine perspective strengths.
    # --------------------------------------------------------

    if vertical_strength is None:
        if actual_mode in (
            "vertical",
            "full",
        ):
            vertical_strength = auto_v
        else:
            vertical_strength = 0.0

    if horizontal_strength is None:
        if actual_mode in (
            "horizontal",
            "full",
        ):
            horizontal_strength = auto_h
        else:
            horizontal_strength = 0.0

    vertical_strength = float(
        np.clip(
            vertical_strength,
            -100.0,
            100.0,
        )
    )

    horizontal_strength = float(
        np.clip(
            horizontal_strength,
            -100.0,
            100.0,
        )
    )

    # Automatic mode gets an additional conservative cap.
    if mode == "auto":
        vertical_strength = float(
            np.clip(
                vertical_strength,
                -AUTO_MAX_PERSPECTIVE,
                AUTO_MAX_PERSPECTIVE,
            )
        )

        horizontal_strength = float(
            np.clip(
                horizontal_strength,
                -AUTO_MAX_PERSPECTIVE,
                AUTO_MAX_PERSPECTIVE,
            )
        )

    # --------------------------------------------------------
    # Build transform.
    # --------------------------------------------------------

    H = np.eye(
        3,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Perspective.
    # --------------------------------------------------------

    if actual_mode == "vertical":
        Hp = vp_projective_transform(
            vertical_strength,
            0.0,
            w,
            h,
            vertical_vp=info.vertical_vp,
            vertical_reference=auto_v,
        )

        H = Hp @ H

    elif actual_mode == "horizontal":
        Hp = vp_projective_transform(
            0.0,
            horizontal_strength,
            w,
            h,
            horizontal_vp=info.horizontal_vp,
            horizontal_reference=auto_h,
        )

        H = Hp @ H

    elif actual_mode == "full":
        Hp = vp_projective_transform(
            vertical_strength,
            horizontal_strength,
            w,
            h,
            vertical_vp=info.vertical_vp,
            horizontal_vp=info.horizontal_vp,
            vertical_reference=auto_v,
            horizontal_reference=auto_h,
        )

        H = Hp @ H

    # --------------------------------------------------------
    # Automatic level.
    #
    # All automatic modes retain level correction.
    # --------------------------------------------------------

    if mode in automatic_modes:
        level = float(
            np.clip(
                info.level,
                -AUTO_MAX_ROTATION,
                AUTO_MAX_ROTATION,
            )
        )

        # A detected positive line angle needs the corresponding
        # OpenCV rotation to flatten it.
        Hr = rotation_homography(
            level,
            w,
            h,
        )

        H = Hr @ H

    # --------------------------------------------------------
    # Manual rotation.
    # --------------------------------------------------------

    if abs(float(rotate)) > 1e-8:
        Hr = rotation_homography(
            float(rotate),
            w,
            h,
        )

        H = Hr @ H

    # --------------------------------------------------------
    # Aspect.
    # --------------------------------------------------------

    if abs(float(aspect)) > 1e-8:
        Ha = aspect_homography(
            float(aspect),
            w,
            h,
        )

        H = Ha @ H

    # --------------------------------------------------------
    # Scale.
    # --------------------------------------------------------

    if abs(float(scale) - 100.0) > 1e-8:
        Hs = scale_homography(
            float(scale),
            w,
            h,
        )

        H = Hs @ H

    # --------------------------------------------------------
    # Offset.
    # --------------------------------------------------------

    if (
        abs(float(x)) > 1e-8
        or abs(float(y)) > 1e-8
    ):
        Ho = offset_homography(
            float(x),
            float(y),
            w,
            h,
        )

        H = Ho @ H

    validate_homography(H)

    return H, info


# ============================================================
# TRANSFORMED CORNERS
# ============================================================

def transformed_corners(
    width: int,
    height: int,
    H: np.ndarray,
) -> np.ndarray:
    """
    Transform the four source-image corners.
    """
    corners = np.array(
        [
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ]
        ],
        dtype=np.float64,
    )

    transformed = cv2.perspectiveTransform(
        corners,
        H,
    )

    result = transformed.reshape(
        -1,
        2,
    )

    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Transform produced invalid corner coordinates."
        )

    return result


# ============================================================
# FIT CANVAS
# ============================================================

def fit_to_canvas(
    H: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Shift transformed content so that all source corners fit
    into a positive output canvas.

    Unlike the previous implementation, this function refuses
    pathological dimensions instead of silently clipping the
    requested canvas while leaving the matrix unchanged.
    """
    pts = transformed_corners(
        width,
        height,
        H,
    )

    min_x = float(
        np.min(pts[:, 0])
    )

    min_y = float(
        np.min(pts[:, 1])
    )

    max_x = float(
        np.max(pts[:, 0])
    )

    max_y = float(
        np.max(pts[:, 1])
    )

    if not all(
        math.isfinite(value)
        for value in (
            min_x,
            min_y,
            max_x,
            max_y,
        )
    ):
        raise RuntimeError(
            "Invalid geometry transform."
        )

    out_w_float = (
        max_x
        - min_x
        + 1.0
    )

    out_h_float = (
        max_y
        - min_y
        + 1.0
    )

    max_w = max(
        MIN_OUTPUT_SIZE,
        int(
            math.ceil(
                width * MAX_OUTPUT_MULTIPLIER
            )
        ),
    )

    max_h = max(
        MIN_OUTPUT_SIZE,
        int(
            math.ceil(
                height * MAX_OUTPUT_MULTIPLIER
            )
        ),
    )

    if (
        out_w_float > max_w
        or out_h_float > max_h
    ):
        raise RuntimeError(
            "Perspective correction produced an "
            "unreasonably large output canvas. "
            "Reduce the correction strength."
        )

    out_w = max(
        MIN_OUTPUT_SIZE,
        int(math.ceil(out_w_float)),
    )

    out_h = max(
        MIN_OUTPUT_SIZE,
        int(math.ceil(out_h_float)),
    )

    shift = translation(
        -min_x,
        -min_y,
    )

    Hfit = shift @ H

    # Perspective rectification can reduce the bounding box dramatically.
    # Preserve the source resolution so the returned image is not a tiny,
    # over-compressed version of the original.
    resolution_scale = max(
        1.0,
        width / max(1.0, out_w),
        height / max(1.0, out_h),
    )

    if resolution_scale > 1.0:
        Hfit = (
            np.array(
                [
                    [resolution_scale, 0.0, 0.0],
                    [0.0, resolution_scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            @ Hfit
        )
        out_w = int(math.ceil(out_w * resolution_scale))
        out_h = int(math.ceil(out_h * resolution_scale))

    validate_homography(Hfit)

    return (
        Hfit,
        out_w,
        out_h,
    )


# ============================================================
# VALIDITY MASK
# ============================================================

def render_validity_mask(
    width: int,
    height: int,
    Hfit: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Warp an all-valid source mask.

    This is much safer than looking for black pixels in the
    rendered photograph because the photograph itself may contain
    legitimate black/dark content.
    """
    source_mask = np.full(
        (
            height,
            width,
        ),
        255,
        dtype=np.uint8,
    )

    mask = cv2.warpPerspective(
        source_mask,
        Hfit,
        (
            out_w,
            out_h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Remove tiny slivers caused by projective interpolation.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


# ============================================================
# VALID CROP
# ============================================================

def crop_using_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Crop to the largest reliable rectangle contained inside the
    geometrically valid region.
    """
    if image.size == 0:
        return image

    if mask.size == 0:
        return image

    if mask.shape[:2] != image.shape[:2]:
        raise RuntimeError(
            "Image and validity mask dimensions do not match."
        )

    h, w = mask.shape[:2]

    if h < 20 or w < 20:
        return image

    # Convert to binary.
    binary = (
        mask > 127
    ).astype(
        np.uint8
    )

    # Slight erosion keeps edge interpolation artifacts out.
    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    binary = cv2.erode(
        binary,
        kernel,
        iterations=1,
    )

    if not np.any(binary):
        return image

    # Largest rectangle in binary matrix.
    heights = np.zeros(
        w,
        dtype=np.int32,
    )

    best_area = 0
    best_rect = (
        0,
        0,
        w,
        h,
    )

    for row_index in range(h):
        row = binary[row_index] > 0

        heights[row] += 1
        heights[~row] = 0

        stack: List[int] = []

        for column_index in range(w + 1):
            current = (
                int(heights[column_index])
                if column_index < w
                else 0
            )

            while (
                stack
                and current
                < heights[stack[-1]]
            ):
                top = stack.pop()

                rectangle_height = int(
                    heights[top]
                )

                left = (
                    stack[-1] + 1
                    if stack
                    else 0
                )

                rectangle_width = (
                    column_index
                    - left
                )

                area = (
                    rectangle_height
                    * rectangle_width
                )

                if area > best_area:
                    best_area = area

                    best_rect = (
                        left,
                        row_index
                        - rectangle_height
                        + 1,
                        rectangle_width,
                        rectangle_height,
                    )

            stack.append(
                column_index
            )

    x, y, cw, ch = best_rect

    # If the valid rectangle is too small, retain the complete result
    # rather than destroying the photograph's usable framing.
    if (
        cw < w * 0.60
        or ch < h * 0.60
    ):
        return image

    return image[
        y:y + ch,
        x:x + cw,
    ]


# ============================================================
# DEBUG LINES
# ============================================================

def draw_debug(
    image: np.ndarray,
    info: GeometryEstimate,
    output_path: str,
) -> None:
    """
    Render detected structural lines and VPs.
    """
    debug = image.copy()

    for line in info.lines:
        angle = abs(line.angle)

        if angle <= MIN_LINE_ANGLE:
            color = (
                255,
                0,
                0,
            )

        elif angle >= MAX_LINE_ANGLE:
            color = (
                0,
                255,
                0,
            )

        else:
            color = (
                0,
                0,
                255,
            )

        cv2.line(
            debug,
            (
                int(round(line.x1)),
                int(round(line.y1)),
            ),
            (
                int(round(line.x2)),
                int(round(line.y2)),
            ),
            color,
            2,
            cv2.LINE_AA,
        )

    # Vertical VP.
    if info.vertical_vp is not None:
        x, y = map(
            int,
            np.round(info.vertical_vp),
        )

        cv2.circle(
            debug,
            (x, y),
            16,
            (
                0,
                255,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

        cv2.drawMarker(
            debug,
            (x, y),
            (
                0,
                0,
                0,
            ),
            cv2.MARKER_CROSS,
            30,
            2,
            cv2.LINE_AA,
        )

    # Horizontal VP.
    if info.horizontal_vp is not None:
        x, y = map(
            int,
            np.round(info.horizontal_vp),
        )

        cv2.circle(
            debug,
            (x, y),
            16,
            (
                255,
                255,
                0,
            ),
            -1,
            cv2.LINE_AA,
        )

        cv2.drawMarker(
            debug,
            (x, y),
            (
                0,
                0,
                0,
            ),
            cv2.MARKER_CROSS,
            30,
            2,
            cv2.LINE_AA,
        )

    write_image(
        output_path,
        debug,
        95,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic Lightroom-style "
            "photo geometry correction."
        )
    )

    parser.add_argument(
        "input",
        help="Input image",
    )

    parser.add_argument(
        "output",
        help="Output image",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "off",
            "auto",
            "level",
            "vertical",
            "horizontal",
            "full",
        ],
        default="auto",
    )

    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help=(
            "Additional manual rotation in degrees."
        ),
    )

    parser.add_argument(
        "--vertical-strength",
        type=float,
        default=None,
        help=(
            "Vertical perspective strength "
            "-100..100."
        ),
    )

    parser.add_argument(
        "--horizontal-strength",
        type=float,
        default=None,
        help=(
            "Horizontal perspective strength "
            "-100..100."
        ),
    )

    parser.add_argument(
        "--aspect",
        type=float,
        default=0.0,
        help="Horizontal aspect -100..100.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale percentage.",
    )

    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Horizontal offset percentage.",
    )

    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="Vertical offset percentage.",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Maximum structural lines to retain.",
    )

    parser.add_argument(
        "--analysis-size",
        type=int,
        default=DEFAULT_ANALYSIS_SIZE,
        help="Maximum analysis dimension.",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop projective borders.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality.",
    )

    parser.add_argument(
        "--debug-lines",
        metavar="PATH",
        default=None,
        help="Write detected-line debug image.",
    )

    parser.add_argument(
        "--print-transform",
        action="store_true",
        help="Print final 3x3 homography.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    try:
        print(
            "Loading image...",
            flush=True,
        )

        image = read_image(
            args.input
        )

        h, w = image.shape[:2]

        print(
            f"Image: {w}x{h}",
            flush=True,
        )

        # ----------------------------------------------------
        # OFF
        # ----------------------------------------------------

        if args.mode == "off":
            write_image(
                args.output,
                image,
                args.quality,
            )

            print(
                f"Saved: {args.output}"
            )

            return 0

        # ----------------------------------------------------
        # ANALYSE + BUILD
        # ----------------------------------------------------

        # The analysis-size CLI argument is intentionally not
        # passed directly to build_transform because its API is
        # kept compatible with the rest of the application.
        H, info = build_transform(
            image,
            mode=args.mode,
            rotate=args.rotate,
            vertical_strength=args.vertical_strength,
            horizontal_strength=args.horizontal_strength,
            aspect=args.aspect,
            scale=args.scale,
            x=args.x,
            y=args.y,
            max_lines=max(
                20,
                args.max_lines,
            ),
            verbose=True,
        )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        if args.debug_lines:
            print(
                "[debug] Writing geometry debug image...",
                flush=True,
            )

            draw_debug(
                image,
                info,
                args.debug_lines,
            )

        # ----------------------------------------------------
        # MATRIX
        # ----------------------------------------------------

        if args.print_transform:
            np.set_printoptions(
                precision=8,
                suppress=True,
            )

            print(
                "Transform:"
            )

            print(H)

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        print(
            "[4/4] Rendering final image...",
            flush=True,
        )

        Hfit, out_w, out_h = fit_to_canvas(
            H,
            w,
            h,
        )

        if (
            out_w < MIN_OUTPUT_SIZE
            or out_h < MIN_OUTPUT_SIZE
        ):
            raise RuntimeError(
                "Geometry produced an invalid "
                "output size."
            )

        # ----------------------------------------------------
        # RENDER
        # ----------------------------------------------------

        result = cv2.warpPerspective(
            image,
            Hfit,
            (
                out_w,
                out_h,
            ),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
            ),
        )

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        if not args.no_crop:
            valid_mask = render_validity_mask(
                w,
                h,
                Hfit,
                out_w,
                out_h,
            )

            result = crop_using_mask(
                result,
                valid_mask,
            )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        if (
            result is None
            or result.size == 0
            or result.ndim != 3
        ):
            raise RuntimeError(
                "Geometry correction produced "
                "an invalid image."
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        write_image(
            args.output,
            result,
            args.quality,
        )

        print(
            f"Saved: {args.output}"
        )

        print(
            f"Output: "
            f"{result.shape[1]}x"
            f"{result.shape[0]}"
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nCancelled."
        )

        return 130

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )