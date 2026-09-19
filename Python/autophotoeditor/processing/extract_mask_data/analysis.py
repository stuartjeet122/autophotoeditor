from __future__ import annotations

from .shared import *
from .masks import *

@dataclass
class WhiteBalanceEstimate:
    temperature: float
    tint: float

    red_factor: float
    green_factor: float
    blue_factor: float

    lab_a: float
    lab_b: float

    confidence: float


def estimate_white_balance(
    data: ImageColorData,
    mask: np.ndarray,
) -> WhiteBalanceEstimate:

    valid = (
        mask > 0.15
    )

    rgb = data.linear_rgb[
        valid
    ]

    lab = data.lab[
        valid
    ]

    if rgb.shape[0] < 20:

        return WhiteBalanceEstimate(
            temperature=0.0,
            tint=0.0,
            red_factor=1.0,
            green_factor=1.0,
            blue_factor=1.0,
            lab_a=0.0,
            lab_b=0.0,
            confidence=0.0,
        )

    # Robust channel means.
    r = robust_mean(
        rgb[:, 0]
    )

    g = robust_mean(
        rgb[:, 1]
    )

    b = robust_mean(
        rgb[:, 2]
    )

    mean_rgb = (
        r + g + b
    ) / 3.0

    r_ratio = safe_ratio(
        r,
        mean_rgb,
    )

    g_ratio = safe_ratio(
        g,
        mean_rgb,
    )

    b_ratio = safe_ratio(
        b,
        mean_rgb,
    )

    # --------------------------------------------------------
    # Neutrality error.
    #
    # Blue/yellow is estimated from B relative to R+G.
    # Green/magenta is estimated from G relative to R+B.
    # --------------------------------------------------------

    blue_error = (
        b_ratio
        -
        (
            r_ratio
            +
            g_ratio
        ) * 0.5
    )

    green_error = (
        g_ratio
        -
        (
            r_ratio
            +
            b_ratio
        ) * 0.5
    )

    # Lab gives an independent chromaticity measurement.
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

    # Do not aggressively white-balance strongly colored objects.
    neutrality = float(
        np.clip(
            1.0
            -
            chroma / 65.0,
            0.15,
            1.0,
        )
    )

    # Temperature:
    # positive = warm correction
    # negative = cool correction.
    temperature = (
        -blue_error
        * 95.0
        *
        neutrality
    )

    # Tint:
    # positive = magenta
    # negative = green.
    tint = (
        green_error
        * 80.0
        +
        mean_a
        * 0.15
    )

    temperature *= 0.55
    tint *= 0.45

    temperature = clip(
        temperature,
        -20.0,
        20.0,
    )

    tint = clip(
        tint,
        -15.0,
        15.0,
    )

    # Normalize toward neutral.
    target = (
        r + g + b
    ) / 3.0

    red_factor = target / max(
        r,
        EPS,
    )

    green_factor = target / max(
        g,
        EPS,
    )

    blue_factor = target / max(
        b,
        EPS,
    )

    confidence = float(
        np.clip(
            rgb.shape[0] / 10000.0,
            0.0,
            1.0,
        )
    )

    return WhiteBalanceEstimate(
        temperature=temperature,
        tint=tint,

        red_factor=red_factor,
        green_factor=green_factor,
        blue_factor=blue_factor,

        lab_a=mean_a,
        lab_b=mean_b,

        confidence=confidence,
    )


# ============================================================
# EXPOSURE
# ============================================================

def exposure_from_luminance(
    stats: RegionStats,
) -> float:
    """
    Calculate an EV-style exposure correction.

    Uses median + percentile structure rather than mean only.
    """

    median = max(
        stats.p50,
        0.001,
    )

    # Middle-gray target in linear luminance.
    target = 0.18

    ev = math.log2(
        target / median
    )

    # Median alone is not enough.
    # If highlights are already very high, reduce the
    # amount of positive exposure.
    if stats.p95 > 0.92:
        ev *= 0.55

    if stats.p99 > 0.98:
        ev *= 0.40

    # If very dark, allow a little more lift.
    if stats.p10 < 0.025:
        ev *= 1.10

    return clip(
        ev,
        -1.5,
        1.5,
    )


# ============================================================
# SHADOW / HIGHLIGHT ANALYSIS
# ============================================================

def calculate_tone_adjustments(
    stats: RegionStats,
) -> Dict[str, float]:

    shadows = 0.0
    highlights = 0.0
    whites = 0.0
    blacks = 0.0

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    if stats.shadow_fraction > 0.08:

        amount = (
            stats.shadow_fraction
            -
            0.08
        ) * 260.0

        shadows = clip(
            amount,
            0.0,
            45.0,
        )

    elif stats.p10 < 0.035:

        shadows = clip(
            (
                0.035
                -
                stats.p10
            )
            * 300.0,
            0.0,
            35.0,
        )

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    if stats.highlight_fraction > 0.06:

        amount = (
            stats.highlight_fraction
            -
            0.06
        ) * 250.0

        highlights = -clip(
            amount,
            0.0,
            50.0,
        )

    elif stats.p95 > 0.94:

        highlights = -clip(
            (
                stats.p95
                -
                0.94
            )
            * 250.0,
            0.0,
            35.0,
        )

    # --------------------------------------------------------
    # Whites
    # --------------------------------------------------------

    if stats.p99 > 0.985:

        whites = -clip(
            (
                stats.p99
                -
                0.985
            )
            * 250.0,
            0.0,
            25.0,
        )

    elif stats.p95 < 0.70:

        whites = clip(
            (
                0.70
                -
                stats.p95
            )
            * 55.0,
            0.0,
            20.0,
        )

    # --------------------------------------------------------
    # Blacks
    # --------------------------------------------------------

    if stats.p01 < 0.008:

        blacks = clip(
            (
                0.008
                -
                stats.p01
            )
            * 900.0,
            0.0,
            15.0,
        )

    elif stats.p10 > 0.18:

        blacks = -clip(
            (
                stats.p10
                -
                0.18
            )
            * 80.0,
            0.0,
            15.0,
        )

    return {
        "shadows": shadows,
        "highlights": highlights,
        "whites": whites,
        "blacks": blacks,
    }


# ============================================================
# CONTRAST
# ============================================================

def calculate_contrast(
    stats: RegionStats,
) -> float:

    dynamic_range = (
        stats.p90
        -
        stats.p10
    )

    # Healthy portrait/image region.
    target_range = 0.55

    difference = (
        dynamic_range
        -
        target_range
    )

    correction = (
        difference
        * 80.0
    )

    # Avoid increasing contrast when clipped.
    if stats.highlight_fraction > 0.08:
        correction -= 8.0

    if stats.shadow_fraction > 0.12:
        correction -= 6.0

    return clip(
        correction,
        -25.0,
        25.0,
    )


# ============================================================
# COLOR INTENSITY
# ============================================================

def calculate_color_adjustments(
    stats: RegionStats,
    mask_name: str,
) -> Dict[str, float]:

    saturation = stats.saturation

    # HSV saturation is 0..1.
    if saturation < 0.16:

        saturation_adjustment = clip(
            (
                0.16
                -
                saturation
            )
            * 45.0,
            0.0,
            10.0,
        )

    elif saturation > 0.72:

        saturation_adjustment = -clip(
            (
                saturation
                -
                0.72
            )
            * 40.0,
            0.0,
            15.0,
        )

    else:

        saturation_adjustment = 0.0

    # Vibrance should normally be more conservative
    # than saturation.
    vibrance = (
        -saturation_adjustment
        * 0.45
    )

    # Skin must remain natural.
    if mask_name in {
        "skin",
        "skin_face",
        "skin_neck",
        "face",
        "hands",
        "arms",
    }:
        saturation_adjustment *= 0.35
        vibrance *= 0.25

    return {
        "saturation": clip(
            saturation_adjustment,
            -25.0,
            25.0,
        ),
        "vibrance": clip(
            vibrance,
            -30.0,
            30.0,
        ),
    }


# ============================================================
# MICRO-CONTRAST
# ============================================================

def local_detail_metric(
    luminance: np.ndarray,
    mask: np.ndarray,
) -> float:

    if np.count_nonzero(
        mask > 0.15
    ) < 20:
        return 0.0

    blurred = cv2.GaussianBlur(
        luminance.astype(
            np.float32
        ),
        (0, 0),
        2.0,
    )

    detail = np.abs(
        luminance
        -
        blurred
    )

    values = detail[
        mask > 0.15
    ]

    if values.size == 0:
        return 0.0

    return float(
        np.median(values)
    )


def calculate_clarity(
    data: ImageColorData,
    mask: np.ndarray,
    mask_name: str,
) -> float:

    detail = local_detail_metric(
        data.luminance,
        mask,
    )

    # Natural-detail range.
    target = 0.018

    correction = (
        target
        -
        detail
    ) * 700.0

    correction = clip(
        correction,
        -15.0,
        15.0,
    )

    # Skin should not receive aggressive clarity.
    if mask_name in {
        "skin",
        "skin_face",
        "skin_neck",
        "face",
        "hands",
        "arms",
    }:
        correction *= 0.15

    if mask_name in {
        "eyes",
        "brows",
        "hair",
        "clothing",
        "top",
        "dress",
        "pants",
        "jewelry",
    }:
        correction *= 0.75

    return clip(
        correction,
        -20.0,
        20.0,
    )


# ============================================================
# DEHAZE
# ============================================================

def calculate_dehaze(
    stats: RegionStats,
) -> float:

    # Haze tends to compress dynamic range and increase
    # luminance/chroma ambiguity.
    dynamic_range = (
        stats.p95
        -
        stats.p05
    )

    if dynamic_range < 0.30:

        amount = (
            0.30
            -
            dynamic_range
        ) * 35.0

        return clip(
            amount,
            0.0,
            10.0,
        )

    return 0.0


# ============================================================
# LOCAL CONTEXT CORRECTION
# ============================================================

def context_exposure_adjustment(
    region: RegionStats,
    surrounding: Optional[RegionStats],
) -> float:

    if surrounding is None:
        return 0.0

    region_l = max(
        region.p50,
        0.001,
    )

    surround_l = max(
        surrounding.p50,
        0.001,
    )

    # Difference in stops.
    difference = math.log2(
        region_l
        /
        surround_l
    )

    # We do NOT attempt to make the mask equal to
    # the surrounding region.
    #
    # We only remove extreme discontinuities.
    if difference > 1.0:
        correction = -(
            difference - 1.0
        ) * 0.30

    elif difference < -1.0:
        correction = -(
            difference + 1.0
        ) * 0.30

    else:
        correction = 0.0

    return clip(
        correction,
        -0.30,
        0.30,
    )


# ============================================================
# LOCAL COLOR NEUTRALIZATION
# ============================================================

def context_white_balance(
    region_wb: WhiteBalanceEstimate,
    surrounding_wb: Optional[WhiteBalanceEstimate],
    mask_name: str,
) -> Tuple[float, float]:

    if surrounding_wb is None:
        return (
            region_wb.temperature,
            region_wb.tint,
        )

    # Compare the mask's chromaticity against its immediate
    # environment rather than blindly making the mask gray.
    temperature_difference = (
        region_wb.temperature
        -
        surrounding_wb.temperature
    )

    tint_difference = (
        region_wb.tint
        -
        surrounding_wb.tint
    )

    # Correct only a portion of the difference.
    temperature = (
        -temperature_difference
        * 0.35
    )

    tint = (
        -tint_difference
        * 0.35
    )

    # Skin should retain natural warm chroma.
    if mask_name in {
        "skin",
        "skin_face",
        "skin_neck",
        "face",
        "hands",
        "arms",
    }:

        temperature *= 0.35
        tint *= 0.35

    return (
        clip(
            temperature,
            -20.0,
            20.0,
        ),
        clip(
            tint,
            -15.0,
            15.0,
        ),
    )


# ============================================================
