from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

from .shared import *


# ============================================================
# MASK PATH / FILE LOADING
# ============================================================

def resolve_mask_path(
    json_path: Path,
    value: str,
) -> Path:
    """
    Resolve a mask path.

    Resolution order:
        1. Absolute path.
        2. Relative to JSON file.
        3. Original relative path.

    The final path is returned even when it does not exist so
    callers can produce a useful warning.
    """

    path = Path(
        str(value)
    ).expanduser()

    if path.is_absolute():
        return path

    candidate = (
        json_path.parent
        /
        path
    )

    if candidate.exists():
        return candidate

    return path


def load_mask_file(
    path: Path,
    shape: Tuple[int, int],
    soft: bool,
) -> Optional[np.ndarray]:
    """
    Load and normalize a mask.

    Hard masks:
        uint8 0/1

    Soft masks:
        float32 0..1
    """

    if not path.exists():
        logger.warning(
            "Mask file does not exist: %s",
            path,
        )
        return None

    try:
        mask = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )
    except Exception as exc:
        logger.warning(
            "Failed reading mask %s: %s",
            path,
            exc,
        )
        return None

    if mask is None:
        logger.warning(
            "Could not decode mask: %s",
            path,
        )
        return None

    if mask.ndim != 2:
        logger.warning(
            "Mask is not single-channel: %s",
            path,
        )
        return None

    h, w = shape

    if mask.shape != (h, w):

        interpolation = (
            cv2.INTER_LINEAR
            if soft
            else cv2.INTER_NEAREST
        )

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=interpolation,
        )

    if soft:

        result = (
            mask.astype(
                np.float32,
                copy=False,
            )
            /
            255.0
        )

        return np.clip(
            result,
            0.0,
            1.0,
        )

    return (
        mask >= 128
    ).astype(
        np.uint8
    )


def load_masks_from_json(
    json_path: Path,
    image_shape: Tuple[int, int],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """
    Load hard and soft masks from the segmentation JSON.

    Invalid individual masks do not abort the complete analysis.
    """

    if not json_path.exists():
        logger.warning(
            "Mask JSON does not exist: %s",
            json_path,
        )

        return {}, {}

    try:

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

    except Exception as exc:

        logger.warning(
            "Could not read mask JSON %s: %s",
            json_path,
            exc,
        )

        return {}, {}

    if not isinstance(
        data,
        dict,
    ):
        logger.warning(
            "Mask JSON root must be an object: %s",
            json_path,
        )

        return {}, {}

    masks = data.get(
        "masks",
        {},
    )

    if not isinstance(
        masks,
        dict,
    ):
        logger.warning(
            "Invalid 'masks' section: %s",
            json_path,
        )

        return {}, {}

    hard_masks: Dict[
        str,
        np.ndarray,
    ] = {}

    soft_masks: Dict[
        str,
        np.ndarray,
    ] = {}

    for name, item in masks.items():

        if not isinstance(
            name,
            str,
        ):
            continue

        if not isinstance(
            item,
            dict,
        ):
            continue

        binary_file = item.get(
            "binary_file"
        )

        soft_file = item.get(
            "soft_file"
        )

        if binary_file:

            path = resolve_mask_path(
                json_path,
                str(binary_file),
            )

            hard = load_mask_file(
                path,
                image_shape,
                soft=False,
            )

            if hard is not None:
                hard_masks[name] = hard

        if soft_file:

            path = resolve_mask_path(
                json_path,
                str(soft_file),
            )

            soft = load_mask_file(
                path,
                image_shape,
                soft=True,
            )

            if soft is not None:
                soft_masks[name] = soft

    return (
        hard_masks,
        soft_masks,
    )


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_mask_for_image(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    name: str = "mask",
) -> np.ndarray:
    """
    Validate and normalize a mask.

    Returned mask:
        float32 0..1
    """

    if mask is None:
        raise ValueError(
            f"{name} is None."
        )

    mask = np.asarray(
        mask
    )

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be 2D."
        )

    if mask.shape != image_shape:
        raise ValueError(
            f"{name} shape {mask.shape} does not match "
            f"image shape {image_shape}."
        )

    if not np.all(
        np.isfinite(mask)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return np.clip(
        mask.astype(
            np.float32,
            copy=False,
        ),
        0.0,
        1.0,
    )


# ============================================================
# MASK SAMPLING
# ============================================================

def get_weighted_pixels(
    array: np.ndarray,
    mask: np.ndarray,
    max_pixels: int = 250_000,
    seed: int = 17,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Extract pixels using a soft mask.

    Unlike the previous implementation, sampling is random but
    deterministic rather than stride-based.

    This avoids systematic spatial bias.
    """

    if array.ndim not in (2, 3):
        raise ValueError(
            "array must be 2D or 3D."
        )

    mask = validate_mask_for_image(
        mask,
        array.shape[:2],
    )

    if max_pixels <= 0:
        raise ValueError(
            "max_pixels must be greater than zero."
        )

    flat_weights = mask.reshape(
        -1
    )

    valid = (
        flat_weights > 0.01
    )

    indices = np.flatnonzero(
        valid
    )

    if indices.size == 0:

        empty_values = np.empty(
            (0,)
            if array.ndim == 2
            else (0, array.shape[2]),
            dtype=array.dtype,
        )

        return (
            empty_values,
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    if indices.size > max_pixels:

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            indices,
            size=max_pixels,
            replace=False,
        )

    flat_array = array.reshape(
        array.shape[0] * array.shape[1],
        *array.shape[2:],
    )

    return (
        flat_array[indices],
        flat_weights[indices].astype(
            np.float32,
            copy=False,
        ),
    )


def mask_values(
    values: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Extract values covered by a mask.

    For soft masks, pixels above threshold are included.
    """

    values = np.asarray(
        values
    )

    mask = validate_mask_for_image(
        mask,
        values.shape[:2],
    )

    valid = (
        mask > threshold
    )

    if values.ndim == 2:
        return values[valid]

    return values[valid]


def weighted_mask_statistics(
    values: np.ndarray,
    mask: np.ndarray,
    max_pixels: int = 250_000,
) -> Dict[str, float]:
    """
    Calculate robust statistics using segmentation confidence
    as statistical weights.
    """

    sampled, weights = get_weighted_pixels(
        values,
        mask,
        max_pixels=max_pixels,
    )

    if sampled.size == 0:

        return {
            "count": 0,
            "effective_pixels": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }

    if sampled.ndim > 1:

        # Statistics are intended for scalar arrays.
        raise ValueError(
            "weighted_mask_statistics expects scalar values."
        )

    effective_pixels = float(
        np.sum(weights)
    )

    return {
        "count": int(
            sampled.size
        ),

        "effective_pixels":
            effective_pixels,

        "mean":
            weighted_mean(
                sampled,
                weights,
            ),

        "median":
            weighted_percentile(
                sampled,
                weights,
                50.0,
            ),

        "p05":
            weighted_percentile(
                sampled,
                weights,
                5.0,
            ),

        "p25":
            weighted_percentile(
                sampled,
                weights,
                25.0,
            ),

        "p75":
            weighted_percentile(
                sampled,
                weights,
                75.0,
            ),

        "p95":
            weighted_percentile(
                sampled,
                weights,
                95.0,
            ),
    }


# ============================================================
# LOCAL SURROUNDING MASK
# ============================================================

def create_surrounding_mask(
    mask: np.ndarray,
    radius: int,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    Create an outer ring around a mask.

    The original mask is excluded from the returned ring.
    """

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    binary = (
        mask >= threshold
    ).astype(
        np.uint8
    )

    if not np.any(binary):
        return np.zeros_like(
            binary
        )

    radius = max(
        1,
        int(radius),
    )

    kernel_size = (
        radius * 2
        +
        1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    dilated = cv2.dilate(
        binary,
        kernel,
    )

    ring = (
        (dilated > 0)
        &
        (binary == 0)
    )

    return ring.astype(
        np.uint8
    )


def surrounding_ring(
    mask: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Generate an adaptive local-surrounding ring.

    Radius is based on the equivalent object size but is
    deliberately constrained to avoid huge rings.
    """

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if mask.ndim != 2:
        raise ValueError(
            "mask must be 2D."
        )

    if image_shape is None:
        image_shape = mask.shape

    if mask.shape != image_shape:
        raise ValueError(
            "mask shape does not match image_shape."
        )

    binary = (
        mask > 0.15
    )

    area = int(
        np.count_nonzero(
            binary
        )
    )

    if area <= 0:
        return np.zeros(
            image_shape,
            dtype=np.uint8,
        )

    equivalent_radius = math.sqrt(
        area / math.pi
    )

    radius = int(
        np.clip(
            equivalent_radius * 0.10,
            4,
            32,
        )
    )

    ring = create_surrounding_mask(
        mask,
        radius,
    )

    # Very thin / small regions can produce a tiny ring.
    if np.count_nonzero(ring) < 20:

        ring = create_surrounding_mask(
            mask,
            max(
                radius,
                12,
            ),
        )

    return ring


def surrounding_soft_ring(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Create a soft falloff ring.

    Useful when matching skin, clothing or subject exposure
    to the surrounding environment.

    Output:
        float32 0..1
    """

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    binary = (
        mask > 0.15
    ).astype(
        np.uint8
    )

    if not np.any(binary):
        return np.zeros_like(
            mask,
            dtype=np.float32,
        )

    radius = max(
        1,
        int(radius),
    )

    kernel_size = (
        radius * 2
        +
        1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    distance = cv2.distanceTransform(
        (
            1 - binary
        ).astype(
            np.uint8
        ),
        cv2.DIST_L2,
        3,
    )

    ring = (
        (
            distance > 0
        )
        &
        (
            distance <= radius
        )
    )

    result = np.zeros_like(
        mask,
        dtype=np.float32,
    )

    result[ring] = (
        1.0
        -
        distance[ring]
        /
        max(
            float(radius),
            1.0,
        )
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# REGION STATISTICS
# ============================================================

@dataclass
class RegionStats:

    pixels: int

    effective_pixels: float

    mask_coverage: float

    luminance_mean: float
    luminance_median: float

    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    rms_contrast: float

    highlight_fraction: float
    shadow_fraction: float

    lab_L: float
    lab_a: float
    lab_b: float

    chroma: float

    red_green_ratio: float
    blue_yellow_ratio: float

    saturation: float


def _weighted_fraction(
    values: np.ndarray,
    weights: np.ndarray,
    condition: np.ndarray,
) -> float:
    """
    Weighted fraction satisfying a condition.
    """

    if values.size == 0:
        return 0.0

    total = float(
        np.sum(weights)
    )

    if total <= EPS:
        return 0.0

    return float(
        np.sum(
            weights[condition]
        )
        /
        total
    )


def _safe_rgb_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted mean of a bounded RGB ratio.

    Extreme ratios are clipped because ratios become unstable
    when the denominator approaches zero.
    """

    denominator = np.maximum(
        denominator,
        EPS,
    )

    ratio = (
        numerator
        /
        denominator
    )

    ratio = np.clip(
        ratio,
        0.0,
        10.0,
    )

    return weighted_mean(
        ratio,
        weights,
    )


def calculate_region_stats(
    data: ImageColorData,
    mask: np.ndarray,
    *,
    max_pixels: int = 250_000,
) -> Optional[RegionStats]:
    """
    Calculate mask-aware region statistics.

    Soft segmentation confidence is used as statistical weight.

    This is substantially more reliable than simply selecting
    mask > 0 pixels.
    """

    image_shape = data.luminance.shape

    mask = validate_mask_for_image(
        mask,
        image_shape,
    )

    active = (
        mask > 0.10
    )

    pixel_count = int(
        np.count_nonzero(
            active
        )
    )

    if pixel_count < 4:
        return None

    lum_values, weights = get_weighted_pixels(
        data.luminance,
        mask,
        max_pixels=max_pixels,
    )

    if lum_values.size < 4:
        return None

    # --------------------------------------------------------
    # LUMINANCE
    # --------------------------------------------------------

    p01 = weighted_percentile(
        lum_values,
        weights,
        1.0,
    )

    p05 = weighted_percentile(
        lum_values,
        weights,
        5.0,
    )

    p10 = weighted_percentile(
        lum_values,
        weights,
        10.0,
    )

    p25 = weighted_percentile(
        lum_values,
        weights,
        25.0,
    )

    p50 = weighted_percentile(
        lum_values,
        weights,
        50.0,
    )

    p75 = weighted_percentile(
        lum_values,
        weights,
        75.0,
    )

    p90 = weighted_percentile(
        lum_values,
        weights,
        90.0,
    )

    p95 = weighted_percentile(
        lum_values,
        weights,
        95.0,
    )

    p99 = weighted_percentile(
        lum_values,
        weights,
        99.0,
    )

    mean = weighted_mean(
        lum_values,
        weights,
    )

    # This is the actual standard deviation of luminance.
    variance = weighted_mean(
        (
            lum_values
            -
            mean
        ) ** 2,
        weights,
    )

    rms_contrast = float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )

    # --------------------------------------------------------
    # HIGHLIGHTS / SHADOWS
    # --------------------------------------------------------

    highlights = _weighted_fraction(
        lum_values,
        weights,
        lum_values >= 0.85,
    )

    shadows = _weighted_fraction(
        lum_values,
        weights,
        lum_values <= 0.10,
    )

    # --------------------------------------------------------
    # LAB
    # --------------------------------------------------------

    lab_values, lab_weights = get_weighted_pixels(
        data.lab,
        mask,
        max_pixels=max_pixels,
        seed=23,
    )

    if lab_values.size == 0:
        return None

    mean_L = weighted_mean(
        lab_values[:, 0],
        lab_weights,
    )

    mean_a = weighted_mean(
        lab_values[:, 1],
        lab_weights,
    )

    mean_b = weighted_mean(
        lab_values[:, 2],
        lab_weights,
    )

    chroma_values = np.sqrt(
        lab_values[:, 1] ** 2
        +
        lab_values[:, 2] ** 2
    )

    chroma = weighted_mean(
        chroma_values,
        lab_weights,
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_values, rgb_weights = get_weighted_pixels(
        data.linear_rgb,
        mask,
        max_pixels=max_pixels,
        seed=31,
    )

    if rgb_values.size == 0:
        return None

    red = rgb_values[:, 0]
    green = rgb_values[:, 1]
    blue = rgb_values[:, 2]

    # These are deliberately calculated as ratios of robust
    # weighted means instead of mean(pixel ratios).
    mean_red = weighted_mean(
        red,
        rgb_weights,
    )

    mean_green = weighted_mean(
        green,
        rgb_weights,
    )

    mean_blue = weighted_mean(
        blue,
        rgb_weights,
    )

    red_green_ratio = safe_ratio(
        mean_red,
        mean_green,
        default=1.0,
    )

    yellow_reference = (
        0.5 * mean_red
        +
        0.5 * mean_green
    )

    blue_yellow_ratio = safe_ratio(
        mean_blue,
        yellow_reference,
        default=1.0,
    )

    # --------------------------------------------------------
    # SATURATION
    # --------------------------------------------------------

    if data.hsv is not None:

        hsv_values, hsv_weights = get_weighted_pixels(
            data.hsv,
            mask,
            max_pixels=max_pixels,
            seed=41,
        )

        saturation = weighted_mean(
            hsv_values[:, 1],
            hsv_weights,
        )

    else:

        # Derive a colorfulness proxy from Lab chroma.
        # This avoids requiring HSV.
        saturation = float(
            np.clip(
                chroma / 100.0,
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    coverage = float(
        np.mean(
            active
        )
    )

    effective_pixels = float(
        np.sum(
            mask[active]
        )
    )

    return RegionStats(
        pixels=pixel_count,

        effective_pixels=effective_pixels,

        mask_coverage=coverage,

        luminance_mean=mean,
        luminance_median=p50,

        p01=p01,
        p05=p05,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,

        rms_contrast=rms_contrast,

        highlight_fraction=highlights,
        shadow_fraction=shadows,

        lab_L=mean_L,
        lab_a=mean_a,
        lab_b=mean_b,

        chroma=chroma,

        red_green_ratio=red_green_ratio,
        blue_yellow_ratio=blue_yellow_ratio,

        saturation=saturation,
    )


# ============================================================
# GLOBAL EXPOSURE ANALYSIS
# ============================================================

@dataclass
class GlobalSceneStats:

    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    mean: float

    rms: float

    clipping_black: float
    clipping_white: float

    dynamic_range: float


def calculate_global_stats(
    data: ImageColorData,
    max_pixels: int = 250_000,
) -> GlobalSceneStats:
    """
    Calculate robust global scene statistics.

    Sampling is deterministic to keep analysis repeatable.
    """

    lum = data.luminance

    values = sample_values(
        lum,
        max_samples=max_pixels,
        seed=101,
    )

    values = finite_values(
        values
    )

    if values.size == 0:

        return GlobalSceneStats(
            p01=0.0,
            p05=0.0,
            p10=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            mean=0.0,
            rms=0.0,
            clipping_black=0.0,
            clipping_white=0.0,
            dynamic_range=0.0,
        )

    p01 = robust_percentile(
        values,
        1,
    )

    p05 = robust_percentile(
        values,
        5,
    )

    p10 = robust_percentile(
        values,
        10,
    )

    p25 = robust_percentile(
        values,
        25,
    )

    p50 = robust_percentile(
        values,
        50,
    )

    p75 = robust_percentile(
        values,
        75,
    )

    p90 = robust_percentile(
        values,
        90,
    )

    p95 = robust_percentile(
        values,
        95,
    )

    p99 = robust_percentile(
        values,
        99,
    )

    mean = float(
        np.mean(values)
    )

    rms = float(
        np.std(values)
    )

    clipping_black = float(
        np.mean(
            values <= 0.005
        )
    )

    clipping_white = float(
        np.mean(
            values >= 0.995
        )
    )

    dynamic_range = float(
        max(
            p99 - p01,
            0.0,
        )
    )

    return GlobalSceneStats(
        p01=p01,
        p05=p05,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,

        mean=mean,
        rms=rms,

        clipping_black=clipping_black,
        clipping_white=clipping_white,

        dynamic_range=dynamic_range,
    )