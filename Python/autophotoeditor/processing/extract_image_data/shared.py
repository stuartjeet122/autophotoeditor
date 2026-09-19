from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-8

SCHEMA_VERSION = "5.0"

MAX_ANALYSIS_DIM = 1600
MAX_ANALYSIS_PIXELS = 300_000
MAX_OPTIMIZATION_PIXELS = 250_000
MAX_WB_PIXELS = 180_000

MIN_WB_GAIN = 0.70
MAX_WB_GAIN = 1.45

LAB_L_SCALE = 100.0 / 255.0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class AnalysisConfig:
    """
    Central configuration for image analysis.

    Keeping thresholds here makes the analyzer deterministic and much easier
    to tune without scattering magic numbers throughout the code.
    """

    max_analysis_dim: int = MAX_ANALYSIS_DIM
    max_analysis_pixels: int = MAX_ANALYSIS_PIXELS
    max_wb_pixels: int = MAX_WB_PIXELS

    min_wb_gain: float = MIN_WB_GAIN
    max_wb_gain: float = MAX_WB_GAIN

    # Pixels below this luminance are treated as unreliable for WB.
    wb_min_luminance: float = 0.035

    # Pixels above this luminance are treated as unreliable for WB.
    wb_max_luminance: float = 0.97

    # Maximum chroma for a pixel to be considered potentially neutral.
    wb_max_chroma: float = 0.12

    # Minimum number of pixels required for a meaningful WB estimate.
    wb_min_samples: int = 300

    # Strength with which robust outlier rejection is applied.
    robust_sigma: float = 2.5


DEFAULT_CONFIG = AnalysisConfig()


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def clamp(
    value: float,
    lo: float,
    hi: float,
) -> float:
    """
    Clamp a scalar safely.
    """
    value = safe_float(value)

    if lo > hi:
        lo, hi = hi, lo

    return float(max(lo, min(hi, value)))


def clamp_array(
    value: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Vectorized finite-safe clamp.
    """
    arr = np.asarray(value, dtype=np.float32)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=hi,
        neginf=lo,
    )

    return np.clip(arr, lo, hi)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError, OverflowError):
        pass

    return float(default)


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (TypeError, ValueError, OverflowError):
        return int(default)


def finite_array(
    values: np.ndarray,
) -> np.ndarray:
    """
    Return only finite values.

    Works for both scalar arrays and NxC arrays.
    """

    arr = np.asarray(values)

    if arr.size == 0:
        return arr

    if arr.ndim == 1:
        return arr[np.isfinite(arr)]

    finite = np.all(np.isfinite(arr), axis=-1)

    return arr[finite]


def percentile(
    values: np.ndarray,
    q: float,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    arr = finite_array(values)

    if arr.size == 0:
        return default

    q = clamp(q, 0.0, 100.0)

    return safe_float(
        np.percentile(arr, q),
        default,
    )


def robust_median(
    values: np.ndarray,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    arr = finite_array(values)

    if arr.size == 0:
        return default

    return safe_float(
        np.median(arr),
        default,
    )


def robust_mean(
    values: np.ndarray,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    arr = finite_array(values)

    if arr.size == 0:
        return default

    return safe_float(
        np.mean(arr),
        default,
    )


def robust_std(
    values: np.ndarray,
    default: float = 0.0,
) -> float:

    if values is None:
        return default

    arr = finite_array(values)

    if arr.size == 0:
        return default

    return safe_float(
        np.std(arr),
        default,
    )


def safe_log(
    value: float,
) -> float:

    return math.log(max(EPS, safe_float(value, EPS)))


# ============================================================================
# IMAGE VALIDATION
# ============================================================================

def validate_bgr_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize an OpenCV BGR image.

    Expected input:
        H x W x 3
        uint8

    A contiguous uint8 copy/view is returned.
    """

    if image is None:
        raise ValueError("Image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be numpy.ndarray, got {type(image)!r}."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Expected HxWx3 image, got ndim={image.ndim}."
        )

    if image.shape[2] != 3:
        raise ValueError(
            f"Expected 3 channels, got {image.shape[2]}."
        )

    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(
            "Image is too small for analysis."
        )

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            image = np.clip(
                image,
                0.0,
                255.0,
            ).astype(np.uint8)

        else:
            image = np.clip(
                image,
                0,
                255,
            ).astype(np.uint8)

    return np.ascontiguousarray(image)


# ============================================================================
# MASK HANDLING
# ============================================================================

def normalized_mask(
    mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Normalize a mask to boolean HxW.

    Important:
    A 3-channel mask is accepted only when all channels are effectively
    equivalent. We do NOT blindly convert arbitrary RGB/BGR data to grayscale,
    because doing so can create false mask pixels.
    """

    if mask is None:
        return None

    m = np.asarray(mask)

    if m.ndim == 3:

        if m.shape[2] == 1:
            m = m[:, :, 0]

        elif m.shape[2] == 3:
            c0 = m[:, :, 0]
            c1 = m[:, :, 1]
            c2 = m[:, :, 2]

            if (
                np.array_equal(c0, c1)
                and np.array_equal(c1, c2)
            ):
                m = c0

            else:
                raise ValueError(
                    "3-channel mask contains different channel data. "
                    "Pass a single-channel mask instead."
                )

        else:
            raise ValueError(
                f"Unsupported mask channel count: {m.shape[2]}."
            )

    if m.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got shape={m.shape}."
        )

    return m > 0


def resize_mask(
    mask: Optional[np.ndarray],
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:

    if mask is None:
        return None

    m = normalized_mask(mask)

    if m is None:
        return None

    height, width = shape

    if m.shape == (height, width):
        return m

    resized = cv2.resize(
        m.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


def mask_fraction(
    mask: Optional[np.ndarray],
) -> float:

    if mask is None:
        return 0.0

    m = normalized_mask(mask)

    if m is None or m.size == 0:
        return 0.0

    return float(np.count_nonzero(m) / m.size)


def intersection_fraction(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
) -> float:

    if a is None or b is None:
        return 0.0

    aa = normalized_mask(a)
    bb = normalized_mask(b)

    if aa is None or bb is None:
        return 0.0

    if aa.shape != bb.shape:
        return 0.0

    return float(
        np.count_nonzero(aa & bb) / aa.size
    )


def union_masks(
    *masks: Optional[np.ndarray],
    shape: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:

    valid = []

    for mask in masks:

        if mask is None:
            continue

        m = normalized_mask(mask)

        if m is None:
            continue

        if shape is not None and m.shape != shape:
            m = resize_mask(m, shape)

        if m is not None:
            valid.append(m)

    if not valid:
        return None

    result = np.zeros_like(
        valid[0],
        dtype=bool,
    )

    for m in valid:
        result |= m

    return result


def invert_mask(
    mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:

    if mask is None:
        return None

    return ~normalized_mask(mask)


# ============================================================================
# IMAGE CONVERSION
# ============================================================================

def bgr_to_rgb_float(
    image: np.ndarray,
) -> np.ndarray:

    image = validate_bgr_image(image)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return rgb.astype(
        np.float32
    ) / 255.0


def srgb_to_linear(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert sRGB [0,1] to linear RGB [0,1].

    Explicitly clamps input to avoid invalid values propagating into
    exposure/WB calculations.
    """

    x = clamp_array(
        rgb,
        0.0,
        1.0,
    )

    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power(
            (x + 0.055) / 1.055,
            2.4,
        ),
    ).astype(np.float32)


def linear_to_srgb(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Convert linear RGB to sRGB.

    Output is clipped to the legal display range.
    """

    x = np.maximum(
        np.asarray(rgb, dtype=np.float32),
        0.0,
    )

    result = np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(
            np.maximum(x, 0.0),
            1.0 / 2.4,
        ) - 0.055,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


def rgb_to_linear_luminance(
    rgb_linear: np.ndarray,
) -> np.ndarray:

    rgb = np.asarray(
        rgb_linear,
        dtype=np.float32,
    )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "Expected HxWx3 RGB array."
        )

    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


# ============================================================================
# LAB
# ============================================================================

def bgr_to_lab(
    image: np.ndarray,
) -> np.ndarray:

    image = validate_bgr_image(image)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )


def lab_l_star(
    lab: np.ndarray,
) -> np.ndarray:

    return (
        np.asarray(
            lab[:, :, 0],
            dtype=np.float32,
        ) * LAB_L_SCALE
    )


def lab_a_star(
    lab: np.ndarray,
) -> np.ndarray:

    return (
        np.asarray(
            lab[:, :, 1],
            dtype=np.float32,
        ) - 128.0
    )


def lab_b_star(
    lab: np.ndarray,
) -> np.ndarray:

    return (
        np.asarray(
            lab[:, :, 2],
            dtype=np.float32,
        ) - 128.0
    )


# ============================================================================
# DETERMINISTIC SPATIAL SAMPLING
# ============================================================================

def sample_array(
    array: np.ndarray,
    max_pixels: int,
) -> np.ndarray:
    """
    Deterministically sample approximately max_pixels PIXELS.

    Unlike the original implementation, max_pixels means pixels rather than
    scalar array elements.

    For HxWx3 input the returned array remains Nx3.
    """

    if max_pixels <= 0:
        raise ValueError(
            "max_pixels must be > 0."
        )

    arr = np.asarray(array)

    if arr.size == 0:
        return arr.copy()

    # Scalar/vector data.
    if arr.ndim == 1:

        if arr.size <= max_pixels:
            return arr

        step = max(
            1,
            int(math.ceil(arr.size / max_pixels)),
        )

        return arr[::step]

    # Image-like HxWxC.
    if arr.ndim >= 2:

        height = arr.shape[0]
        width = arr.shape[1]

        pixels = height * width

        if pixels <= max_pixels:
            return arr.reshape(
                pixels,
                *arr.shape[2:],
            )

        # Deterministic approximately-uniform grid.
        scale = math.sqrt(
            pixels / max_pixels
        )

        step_y = max(
            1,
            int(round(scale)),
        )

        step_x = step_y

        sampled = arr[
            ::step_y,
            ::step_x,
        ]

        return sampled.reshape(
            -1,
            *arr.shape[2:],
        )

    return arr.reshape(-1)


def sample_masked(
    array: np.ndarray,
    mask: Optional[np.ndarray],
    max_pixels: int,
) -> np.ndarray:
    """
    Deterministically sample pixels selected by a binary mask.
    """

    arr = np.asarray(array)

    if arr.size == 0:
        return arr.copy()

    if mask is None:
        return sample_array(
            arr,
            max_pixels,
        )

    m = normalized_mask(mask)

    if m is None:
        return sample_array(
            arr,
            max_pixels,
        )

    if arr.ndim < 2:
        raise ValueError(
            "Masked sampling requires an image-like array."
        )

    if m.shape != arr.shape[:2]:
        raise ValueError(
            f"Mask shape {m.shape} does not match "
            f"array shape {arr.shape[:2]}."
        )

    selected = arr[m]

    if selected.shape[0] <= max_pixels:
        return selected

    step = max(
        1,
        int(math.ceil(
            selected.shape[0] / max_pixels
        )),
    )

    return selected[::step]


# ============================================================================
# ROBUST OUTLIER REJECTION
# ============================================================================

def robust_trim(
    values: np.ndarray,
    sigma: float = 2.5,
) -> np.ndarray:
    """
    Remove extreme values using MAD rather than standard deviation.

    MAD is much more stable for photographic distributions containing
    highlights, shadows and clipped pixels.
    """

    arr = finite_array(values)

    if arr.size < 8:
        return arr

    median = np.median(arr)

    mad = np.median(
        np.abs(arr - median)
    )

    if mad < EPS:
        return arr

    robust_sigma = 1.4826 * mad

    keep = np.abs(
        arr - median
    ) <= sigma * robust_sigma

    trimmed = arr[keep]

    return trimmed if trimmed.size else arr


# ============================================================================
# ROBUST RGB STATISTICS
# ============================================================================

def robust_rgb_mean(
    rgb: np.ndarray,
    max_pixels: int = MAX_WB_PIXELS,
) -> Tuple[float, float, float]:

    pixels = sample_array(
        rgb,
        max_pixels,
    )

    pixels = np.asarray(
        pixels,
        dtype=np.float32,
    ).reshape(-1, 3)

    pixels = finite_array(
        pixels
    )

    if pixels.size == 0:
        return (
            1.0,
            1.0,
            1.0,
        )

    result = []

    for channel in range(3):

        values = robust_trim(
            pixels[:, channel]
        )

        result.append(
            robust_mean(
                values,
                1.0,
            )
        )

    return tuple(result)


# ============================================================================
# WHITE-BALANCE HELPERS
# ============================================================================

def rgb_chroma(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Normalized chroma estimate.

    This is intentionally simple and robust for neutral-pixel selection.
    """

    x = np.asarray(
        rgb,
        dtype=np.float32,
    )

    maximum = np.max(
        x,
        axis=-1,
    )

    minimum = np.min(
        x,
        axis=-1,
    )

    denominator = np.maximum(
        maximum,
        EPS,
    )

    return (
        maximum - minimum
    ) / denominator


def build_neutral_mask(
    rgb_linear: np.ndarray,
    luminance: np.ndarray,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Find candidate neutral pixels.

    Important:
    Low saturation alone is NOT enough.

    We require:
      - sufficient exposure
      - no severe highlight clipping
      - low RGB chroma
    """

    rgb = np.asarray(
        rgb_linear,
        dtype=np.float32,
    )

    lum = np.asarray(
        luminance,
        dtype=np.float32,
    )

    chroma = rgb_chroma(rgb)

    finite = np.all(
        np.isfinite(rgb),
        axis=-1,
    ) & np.isfinite(lum)

    return (
        finite
        & (lum >= config.wb_min_luminance)
        & (lum <= config.wb_max_luminance)
        & (chroma <= config.wb_max_chroma)
    )


def estimate_gray_world_gains(
    rgb_linear: np.ndarray,
    candidate_mask: np.ndarray,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Estimate WB gains from candidate neutral pixels.

    Returns both the estimate and confidence metadata.
    """

    selected = rgb_linear[
        candidate_mask
    ]

    selected = np.asarray(
        selected,
        dtype=np.float32,
    ).reshape(-1, 3)

    selected = finite_array(
        selected
    )

    sample_count = int(
        selected.shape[0]
    )

    if sample_count < config.wb_min_samples:

        return {
            "valid": False,
            "confidence": 0.0,
            "sample_count": sample_count,
            "gains": {
                "r": 1.0,
                "g": 1.0,
                "b": 1.0,
            },
        }

    # Robust channel estimates.
    channels = []

    for i in range(3):

        values = robust_trim(
            selected[:, i],
            config.robust_sigma,
        )

        channels.append(
            robust_median(
                values,
                1.0,
            )
        )

    r, g, b = channels

    if (
        r <= EPS
        or g <= EPS
        or b <= EPS
    ):
        return {
            "valid": False,
            "confidence": 0.0,
            "sample_count": sample_count,
            "gains": {
                "r": 1.0,
                "g": 1.0,
                "b": 1.0,
            },
        }

    target = (
        r + g + b
    ) / 3.0

    gains = np.array(
        [
            target / r,
            target / g,
            target / b,
        ],
        dtype=np.float32,
    )

    gains = np.clip(
        gains,
        config.min_wb_gain,
        config.max_wb_gain,
    )

    # Confidence combines sample count and channel agreement.
    ratios = np.array(
        [
            r / g,
            b / g,
        ],
        dtype=np.float32,
    )

    deviation = float(
        np.mean(
            np.abs(
                np.log(
                    np.maximum(
                        ratios,
                        EPS,
                    )
                )
            )
        )
    )

    sample_confidence = clamp(
        sample_count / 5000.0,
        0.0,
        1.0,
    )

    agreement_confidence = math.exp(
        -deviation * 3.0
    )

    confidence = clamp(
        sample_confidence
        * agreement_confidence,
        0.0,
        1.0,
    )

    return {
        "valid": True,
        "confidence": confidence,
        "sample_count": sample_count,
        "gains": {
            "r": float(gains[0]),
            "g": float(gains[1]),
            "b": float(gains[2]),
        },
        "source_rgb": {
            "r": float(r),
            "g": float(g),
            "b": float(b),
        },
    }