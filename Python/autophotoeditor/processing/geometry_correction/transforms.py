from __future__ import annotations

from .core import *
from .detection import *

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
            f"{level:.3f}Â°",
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

    determinant = float(np.linalg.det(rectifier))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
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


