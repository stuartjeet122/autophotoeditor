from __future__ import annotations

from .shared import *

def resolve_mask_path(
    json_path: Path,
    value: str,
) -> Path:

    path = Path(value)

    if path.is_absolute():
        return path

    # First interpret relative to JSON.
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

    if not path.exists():
        logger.warning(
            "Mask file does not exist: %s",
            path,
        )
        return None

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        logger.warning(
            "Could not read mask: %s",
            path,
        )
        return None

    h, w = shape

    if mask.shape != (h, w):

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=(
                cv2.INTER_LINEAR
                if soft
                else cv2.INTER_NEAREST
            ),
        )

    if soft:
        return (
            mask.astype(np.float32)
            /
            255.0
        )

    return (
        mask > 127
    ).astype(np.uint8)


def load_masks_from_json(
    json_path: Path,
    image_shape: Tuple[int, int],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    hard_masks = {}
    soft_masks = {}

    masks = data.get(
        "masks",
        {},
    )

    for name, item in masks.items():

        if not isinstance(item, dict):
            continue

        binary_file = item.get(
            "binary_file"
        )

        soft_file = item.get(
            "soft_file"
        )

        hard = None
        soft = None

        if binary_file:
            hard = load_mask_file(
                resolve_mask_path(
                    json_path,
                    binary_file,
                ),
                image_shape,
                soft=False,
            )

        if soft_file:
            soft = load_mask_file(
                resolve_mask_path(
                    json_path,
                    soft_file,
                ),
                image_shape,
                soft=True,
            )

        if hard is not None:
            hard_masks[name] = hard

        if soft is not None:
            soft_masks[name] = soft

    return (
        hard_masks,
        soft_masks,
    )


# ============================================================
# MASK SAMPLING
# ============================================================

def get_weighted_pixels(
    array: np.ndarray,
    mask: np.ndarray,
    max_pixels: int = 250_000,
) -> Tuple[np.ndarray, np.ndarray]:

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be 2D."
        )

    weights = mask.astype(
        np.float32
    )

    if np.count_nonzero(weights > 0.01) == 0:
        return (
            np.empty(
                (0,),
                dtype=array.dtype,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    flat_weights = weights.ravel()

    valid = (
        flat_weights > 0.01
    )

    indices = np.flatnonzero(
        valid
    )

    if indices.size > max_pixels:

        # Deterministic sampling.
        step = max(
            1,
            indices.size // max_pixels,
        )

        indices = indices[
            ::step
        ][:max_pixels]

    flat_array = array.reshape(
        -1,
        *array.shape[2:],
    )

    return (
        flat_array[indices],
        flat_weights[indices],
    )


def mask_values(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    if values.ndim == 2:
        return values[
            mask > 0
        ]

    return values[
        mask > 0
    ]


# ============================================================
# LOCAL SURROUNDING MASK
# ============================================================

def create_surrounding_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:

    binary = (
        mask > 0.15
    ).astype(np.uint8)

    if np.count_nonzero(binary) == 0:
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
        dilated
        &
        (~binary.astype(bool))
    ).astype(np.uint8)

    return ring


def surrounding_ring(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:

    h, w = image_shape

    area = np.count_nonzero(
        mask > 0.15
    )

    if area <= 0:
        return np.zeros(
            (h, w),
            dtype=np.uint8,
        )

    # Adaptive radius.
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

    # Fallback for extremely thin regions.
    if np.count_nonzero(ring) < 20:
        ring = create_surrounding_mask(
            mask,
            max(radius, 12),
        )

    return ring


# ============================================================
# STATISTICS
# ============================================================

@dataclass
class RegionStats:
    pixels: int

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


def calculate_region_stats(
    data: ImageColorData,
    mask: np.ndarray,
) -> Optional[RegionStats]:

    valid = (
        mask > 0.10
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 4:
        return None

    lum = data.luminance[
        valid
    ]

    lab = data.lab[
        valid
    ]

    rgb = data.linear_rgb[
        valid
    ]

    hsv = data.hsv[
        valid
    ]

    p01 = robust_percentile(
        lum,
        1,
    )

    p05 = robust_percentile(
        lum,
        5,
    )

    p10 = robust_percentile(
        lum,
        10,
    )

    p25 = robust_percentile(
        lum,
        25,
    )

    p50 = robust_percentile(
        lum,
        50,
    )

    p75 = robust_percentile(
        lum,
        75,
    )

    p90 = robust_percentile(
        lum,
        90,
    )

    p95 = robust_percentile(
        lum,
        95,
    )

    p99 = robust_percentile(
        lum,
        99,
    )

    mean = robust_mean(
        lum
    )

    rms = float(
        np.sqrt(
            np.mean(
                (
                    lum
                    -
                    mean
                ) ** 2
            )
        )
    )

    # Linear-light highlight/shadow definitions.
    highlights = float(
        np.mean(
            lum > 0.85
        )
    )

    shadows = float(
        np.mean(
            lum < 0.10
        )
    )

    mean_L = float(
        np.mean(lab[:, 0])
    )

    mean_a = float(
        np.mean(lab[:, 1])
    )

    mean_b = float(
        np.mean(lab[:, 2])
    )

    chroma = float(
        np.mean(
            np.sqrt(
                lab[:, 1] ** 2
                +
                lab[:, 2] ** 2
            )
        )
    )

    red = np.mean(
        rgb[:, 0]
    )

    green = np.mean(
        rgb[:, 1]
    )

    blue = np.mean(
        rgb[:, 2]
    )

    # Red/green indicates magenta/green direction.
    rg = safe_ratio(
        red,
        green,
    )

    # Blue/yellow direction.
    by = safe_ratio(
        blue,
        (
            0.5 * red
            +
            0.5 * green
        ),
    )

    saturation = float(
        np.mean(
            hsv[:, 1]
        )
    )

    return RegionStats(
        pixels=count,

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

        rms_contrast=rms,

        highlight_fraction=highlights,
        shadow_fraction=shadows,

        lab_L=mean_L,
        lab_a=mean_a,
        lab_b=mean_b,

        chroma=chroma,

        red_green_ratio=rg,
        blue_yellow_ratio=by,

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


def calculate_global_stats(
    data: ImageColorData,
) -> GlobalSceneStats:

    lum = data.luminance.ravel()

    return GlobalSceneStats(
        p01=robust_percentile(
            lum,
            1,
        ),
        p05=robust_percentile(
            lum,
            5,
        ),
        p10=robust_percentile(
            lum,
            10,
        ),
        p25=robust_percentile(
            lum,
            25,
        ),
        p50=robust_percentile(
            lum,
            50,
        ),
        p75=robust_percentile(
            lum,
            75,
        ),
        p90=robust_percentile(
            lum,
            90,
        ),
        p95=robust_percentile(
            lum,
            95,
        ),
        p99=robust_percentile(
            lum,
            99,
        ),
        mean=float(
            np.mean(lum)
        ),
        rms=float(
            np.std(lum)
        ),
        clipping_black=float(
            np.mean(lum <= 0.005)
        ),
        clipping_white=float(
            np.mean(lum >= 0.995)
        ),
    )


# ============================================================
