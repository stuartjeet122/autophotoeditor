from __future__ import annotations

from .core import *

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


