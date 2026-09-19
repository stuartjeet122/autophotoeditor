from __future__ import annotations

from .core import *
from .io import *

class ImageAnalysis:

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:

        self.image = image

        self.height, self.width = image.shape[:2]

        self.gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        self.luma = self.gray / 255.0

        self.lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        self.lab_a = (
            self.lab[:, :, 1]
            .astype(np.float32)
        )

        self.lab_b = (
            self.lab[:, :, 2]
            .astype(np.float32)
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        self.saturation = (
            hsv[:, :, 1]
            .astype(np.float32)
            / 255.0
        )

        self.value = (
            hsv[:, :, 2]
            .astype(np.float32)
            / 255.0
        )

        # Full-image Laplacian is calculated only once.
        self.laplacian = cv2.Laplacian(
            self.gray,
            cv2.CV_32F,
        )

    def crop(
        self,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
    ) -> "ImageAnalysis":
        """Create a lightweight view over already-computed analysis arrays."""

        result = object.__new__(ImageAnalysis)
        result.image = self.image[y1:y2, x1:x2]
        result.height = y2 - y1
        result.width = x2 - x1
        result.gray = self.gray[y1:y2, x1:x2]
        result.luma = self.luma[y1:y2, x1:x2]
        result.lab = self.lab[y1:y2, x1:x2]
        result.lab_a = self.lab_a[y1:y2, x1:x2]
        result.lab_b = self.lab_b[y1:y2, x1:x2]
        result.saturation = self.saturation[y1:y2, x1:x2]
        result.value = self.value[y1:y2, x1:x2]
        result.laplacian = self.laplacian[y1:y2, x1:x2]
        return result


# ============================================================================
# STATISTICS
# ============================================================================

def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:

    if values.size == 0:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not np.any(valid):
        return float(
            np.percentile(
                values,
                percentile,
            )
        )

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    target = (
        cumulative[-1]
        * percentile
        / 100.0
    )

    index = int(
        np.searchsorted(
            cumulative,
            target,
        )
    )

    index = max(
        0,
        min(
            index,
            len(values) - 1,
        ),
    )

    return float(values[index])


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:

    if values.size == 0:
        return 0.0

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    total = float(
        np.sum(weights)
    )

    if total <= 1e-8:
        return float(
            np.mean(values)
        )

    return float(
        np.sum(values * weights)
        / total
    )


def analyze_region(
    analysis: ImageAnalysis,
    mask: np.ndarray,
) -> Dict[str, float]:

    valid = mask > 0.05

    if not np.any(valid):
        return {
            "pixels": 0,
            "brightness": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "saturation": 0.0,
            "shadow_fraction": 0.0,
            "highlight_fraction": 0.0,
            "clipped_fraction": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "lab_a": 128.0,
            "lab_b": 128.0,
        }

    weights = mask[valid]

    luma = analysis.luma[valid]

    saturation = analysis.saturation[valid]

    lab_a = analysis.lab_a[valid]
    lab_b = analysis.lab_b[valid]

    lap = analysis.laplacian[valid]

    p10 = weighted_percentile(
        luma,
        weights,
        10,
    )

    p25 = weighted_percentile(
        luma,
        weights,
        25,
    )

    p50 = weighted_percentile(
        luma,
        weights,
        50,
    )

    p75 = weighted_percentile(
        luma,
        weights,
        75,
    )

    p90 = weighted_percentile(
        luma,
        weights,
        90,
    )

    return {
        "pixels": int(
            np.count_nonzero(valid)
        ),

        "brightness": weighted_mean(
            luma,
            weights,
        ),

        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,

        "saturation": weighted_mean(
            saturation,
            weights,
        ),

        "shadow_fraction": float(
            np.mean(luma < 0.10)
        ),

        "highlight_fraction": float(
            np.mean(luma > 0.92)
        ),

        "clipped_fraction": float(
            np.mean(luma > 0.985)
        ),

        "contrast": float(
            p90 - p10
        ),

        "sharpness": float(
            np.var(lap)
        ),

        "lab_a": weighted_mean(
            lab_a,
            weights,
        ),

        "lab_b": weighted_mean(
            lab_b,
            weights,
        ),
    }


# ============================================================================
# BOUNDING BOX
# ============================================================================

def mask_bbox(
    mask: np.ndarray,
    padding: int,
) -> Optional[Tuple[int, int, int, int]]:

    ys, xs = np.where(
        mask > 0.02
    )

    if len(xs) == 0:
        return None

    h, w = mask.shape

    x1 = max(
        0,
        int(xs.min()) - padding,
    )

    y1 = max(
        0,
        int(ys.min()) - padding,
    )

    x2 = min(
        w,
        int(xs.max()) + padding + 1,
    )

    y2 = min(
        h,
        int(ys.max()) + padding + 1,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================================
# CONTEXT / SURROUNDING REGION
# ============================================================================

def surrounding_ring(
    mask: np.ndarray,
    all_masks: Optional[np.ndarray] = None,
    inner_radius: int = 5,
    outer_radius: int = 35,
) -> np.ndarray:

    mask_u8 = np.where(
        mask > 0.05,
        255,
        0,
    ).astype(np.uint8)

    if outer_radius <= 0:
        return np.zeros_like(
            mask,
            dtype=np.float32,
        )

    outer_kernel_size = (
        outer_radius * 2 + 1
    )

    inner_kernel_size = (
        inner_radius * 2 + 1
    )

    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            outer_kernel_size,
            outer_kernel_size,
        ),
    )

    outer = cv2.dilate(
        mask_u8,
        outer_kernel,
    )

    if inner_radius > 0:

        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                inner_kernel_size,
                inner_kernel_size,
            ),
        )

        inner = cv2.dilate(
            mask_u8,
            inner_kernel,
        )

    else:
        inner = mask_u8

    ring = (
        outer.astype(np.float32)
        -
        inner.astype(np.float32)
    )

    ring = np.clip(
        ring / 255.0,
        0.0,
        1.0,
    )

    # Exclude other semantic masks.
    if all_masks is not None:
        ring[
            all_masks > 0.05
        ] = 0.0

    return ring


def analyze_surrounding(
    analysis: ImageAnalysis,
    mask: np.ndarray,
    all_masks: Optional[np.ndarray] = None,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
]:

    ring = surrounding_ring(
        mask,
        all_masks=all_masks,
        inner_radius=CONFIG.ring_inner_radius,
        outer_radius=CONFIG.ring_outer_radius,
    )

    if np.count_nonzero(ring > 0.05) < 50:

        # Fallback to a slightly wider ring.
        ring = surrounding_ring(
            mask,
            all_masks=None,
            inner_radius=3,
            outer_radius=55,
        )

    stats = analyze_region(
        analysis,
        ring,
    )

    return stats, ring


# ============================================================================
# GLOBAL NEUTRAL ILLUMINANT
# ============================================================================

def estimate_global_neutral_gains(
    analysis: ImageAnalysis,
) -> Tuple[float, float, float]:

    image = analysis.image

    # Work on a reduced image for speed.
    h, w = image.shape[:2]

    scale = min(
        1.0,
        900.0 / max(h, w),
    )

    if scale < 1.0:

        small = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:
        small = image

    b = small[:, :, 0].astype(
        np.float32
    )

    g = small[:, :, 1].astype(
        np.float32
    )

    r = small[:, :, 2].astype(
        np.float32
    )

    gray = (
        0.114 * b
        + 0.587 * g
        + 0.299 * r
    )

    mx = np.maximum.reduce(
        [r, g, b]
    )

    mn = np.minimum.reduce(
        [r, g, b]
    )

    chroma_ratio = (
        (mx - mn)
        / np.maximum(mx, 1.0)
    )

    # Neutral-ish, non-extreme pixels.
    valid = (
        (gray > 30)
        & (gray < 235)
        & (chroma_ratio < 0.18)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 100:

        return (
            1.0,
            1.0,
            1.0,
        )

    rv = r[valid]
    gv = g[valid]
    bv = b[valid]

    # Median channel values are more robust than a plain mean.
    rm = float(
        np.median(rv)
    )

    gm = float(
        np.median(gv)
    )

    bm = float(
        np.median(bv)
    )

    avg = (
        rm + gm + bm
    ) / 3.0

    if min(
        rm,
        gm,
        bm,
    ) < 20:
        return (
            1.0,
            1.0,
            1.0,
        )

    r_gain = avg / rm
    g_gain = avg / gm
    b_gain = avg / bm

    r_gain = float(
        np.clip(
            r_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    g_gain = float(
        np.clip(
            g_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    b_gain = float(
        np.clip(
            b_gain,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    return (
        r_gain,
        g_gain,
        b_gain,
    )


# ============================================================================
# LOCAL NEUTRAL COLOR ESTIMATION
# ============================================================================

def estimate_local_neutral_gains(
    analysis: ImageAnalysis,
    ring: np.ndarray,
    global_gains: Tuple[float, float, float],
) -> Tuple[
    Tuple[float, float, float],
    float,
]:

    image = analysis.image

    valid = (
        (ring > 0.05)
        &
        (analysis.luma > 0.12)
        &
        (analysis.luma < 0.90)
        &
        (analysis.saturation < 0.28)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 60:

        return (
            global_gains,
            0.0,
        )

    pixels = image[valid].astype(
        np.float32
    )

    b = pixels[:, 0]
    g = pixels[:, 1]
    r = pixels[:, 2]

    # Reject pixels whose channels are too different.
    mx = np.maximum.reduce(
        [r, g, b]
    )

    mn = np.minimum.reduce(
        [r, g, b]
    )

    ratio = (
        (mx - mn)
        / np.maximum(mx, 1.0)
    )

    good = ratio < 0.16

    if np.count_nonzero(good) < 40:

        return (
            global_gains,
            0.0,
        )

    r = r[good]
    g = g[good]
    b = b[good]

    # Use medians to avoid colored-object contamination.
    rm = float(
        np.median(r)
    )

    gm = float(
        np.median(g)
    )

    bm = float(
        np.median(b)
    )

    avg = (
        rm + gm + bm
    ) / 3.0

    if min(
        rm,
        gm,
        bm,
    ) < 15:

        return (
            global_gains,
            0.0,
        )

    local_r = avg / rm
    local_g = avg / gm
    local_b = avg / bm

    local_r = float(
        np.clip(
            local_r,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    local_g = float(
        np.clip(
            local_g,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    local_b = float(
        np.clip(
            local_b,
            CONFIG.min_channel_gain,
            CONFIG.max_channel_gain,
        )
    )

    # Confidence depends on sample count.
    confidence = float(
        np.clip(
            count / 1000.0,
            0.0,
            1.0,
        )
    )

    # Local correction is deliberately blended with global correction.
    blend = 0.30 * confidence

    result = (
        global_gains[0] * (1.0 - blend)
        + local_r * blend,

        global_gains[1] * (1.0 - blend)
        + local_g * blend,

        global_gains[2] * (1.0 - blend)
        + local_b * blend,
    )

    return (
        result,
        confidence,
    )


