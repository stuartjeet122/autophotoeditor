from __future__ import annotations

from .core import *

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


