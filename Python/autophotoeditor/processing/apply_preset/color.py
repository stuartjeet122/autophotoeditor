from __future__ import annotations

from .core import *

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)

    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4)
        - 0.055,
    ).astype(np.float32)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    RGB sRGB -> OKLab.
    """

    linear = srgb_to_linear(rgb)

    matrix1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float32,
    )

    lms = linear @ matrix1.T

    lms = np.cbrt(
        np.maximum(lms, 0.0)
    )

    matrix2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float32,
    )

    return (lms @ matrix2.T).astype(np.float32)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:

    matrix1 = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ],
        dtype=np.float32,
    )

    lms = lab @ matrix1.T
    lms = lms ** 3

    matrix2 = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ],
        dtype=np.float32,
    )

    linear = lms @ matrix2.T

    return clamp01(
        linear_to_srgb(linear)
    )


def rgb_to_oklch(
    rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    lab = rgb_to_oklab(rgb)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    C = np.sqrt(
        a * a + b * b
    )

    H = np.degrees(
        np.arctan2(b, a)
    )

    H = np.mod(H, 360.0)

    return L, C, H


def oklch_to_rgb(
    L: np.ndarray,
    C: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    radians = np.radians(H)

    a = C * np.cos(radians)
    b = C * np.sin(radians)

    lab = np.stack(
        [L, a, b],
        axis=-1,
    )

    return oklab_to_rgb(lab)


