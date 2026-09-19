from __future__ import annotations

from .core import *

def load_image(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns RGB float32 [0,1] and optional alpha.
    """

    if os.path.splitext(path)[1].lower() in RAW_EXTENSIONS:
        image = cv2.cvtColor(decode_color(Path(path)), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    if image.ndim == 2:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

        rgb = (
            rgb.astype(np.float32) / 255.0
        )

        return rgb, None

    if image.shape[2] == 4:
        bgr = image[..., :3]

        alpha = (
            image[..., 3].astype(np.float32)
            / 255.0
        )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        return (
            rgb.astype(np.float32) / 255.0,
            alpha,
        )

    bgr = image[..., :3]

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        rgb.astype(np.float32) / 255.0,
        None,
    )


def save_image(
    path: str,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    jpeg_quality: int = 97,
):
    rgb = finite_image(rgb)

    rgb8 = np.round(
        rgb * 255.0
    ).astype(np.uint8)

    ext = os.path.splitext(path)[1].lower()

    if alpha is not None:

        alpha8 = np.round(
            np.clip(alpha, 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        rgba = np.dstack(
            [rgb8, alpha8]
        )

        image = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGRA,
        )

    else:

        image = cv2.cvtColor(
            rgb8,
            cv2.COLOR_RGB2BGR,
        )

    params = []

    if ext in {".jpg", ".jpeg"}:
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(np.clip(jpeg_quality, 1, 100)),
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ]

    elif ext == ".png":
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    ok = cv2.imwrite(
        path,
        image,
        params,
    )

    if not ok:
        raise RuntimeError(
            f"Could not write image: {path}"
        )


# ============================================================================
# DEBUG OUTPUT
# ============================================================================

def save_debug_stage(
    image: np.ndarray,
    alpha: Optional[np.ndarray],
    debug_dir: str,
    name: str,
):
    os.makedirs(
        debug_dir,
        exist_ok=True,
    )

    path = os.path.join(
        debug_dir,
        f"{name}.png",
    )

    save_image(
        path,
        image,
        alpha,
    )

    print(
        f"[DEBUG] {path}"
    )


def image_statistics(
    image: np.ndarray,
) -> str:

    return (
        f"min={float(image.min()):.4f}, "
        f"max={float(image.max()):.4f}, "
        f"mean={float(image.mean()):.4f}"
    )


