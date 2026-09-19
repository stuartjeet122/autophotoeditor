from __future__ import annotations

from .core import *
from .transforms import *

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


