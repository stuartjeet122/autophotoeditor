from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .shared import *
from .masks import *


# ============================================================
# CONSTANTS
# ============================================================

WB_MIN_PIXELS = 32

WB_MIN_CONFIDENCE = 0.08

WB_MAX_GAIN = 1.35
WB_MIN_GAIN = 0.74

EXPOSURE_MIN_EV = -1.50
EXPOSURE_MAX_EV = 1.50

TEMPERATURE_LIMIT = 20.0
TINT_LIMIT = 15.0


# ============================================================
# WHITE BALANCE
# ============================================================

@dataclass
class WhiteBalanceEstimate:

    temperature: float
    tint: float

    red_factor: float
    green_factor: float
    blue_factor: float

    lab_a: float
    lab_b: float

    chroma: float

    neutrality: float

    channel_agreement: float

    confidence: float


def _safe_gain(
    target: float,
    value: float,
) -> float:
    """
    Calculate a bounded channel gain.

    Prevents dark/noisy regions from generating extreme WB
    multipliers.
    """

    if not np.isfinite(
        target
    ):
        return 1.0

    if not np.isfinite(
        value
    ):
        return 1.0

    if value <= EPS:
        return 1.0

    gain = (
        target
        /
        value
    )

    return clip(
        gain,
        WB_MIN_GAIN,
        WB_MAX_GAIN,
    )


def _channel_balance_error(
    r: float,
    g: float,
    b: float,
) -> Tuple[
    float,
    float,
]:
    """
    Estimate chromatic imbalance.

    Returns:
        blue_yellow_error
        green_magenta_error

    Positive blue error:
        image is relatively blue.

    Positive green error:
        image is relatively green.
    """

    total = (
        r + g + b
    )

    if total <= EPS:
        return (
            0.0,
            0.0,
        )

    r_n = r / total
    g_n = g / total
    b_n = b / total

    # Expected neutral proportions.
    neutral = 1.0 / 3.0

    blue_error = (
        b_n
        -
        neutral
    )

    # Green relative to red + blue.
    green_reference = (
        r_n + b_n
    ) * 0.5

    green_error = (
        g_n
        -
        green_reference
    )

    return (
        float(blue_error),
        float(green_error),
    )


def _wb_neutrality(
    chroma: float,
    lab_a: float,
    lab_b: float,
) -> float:
    """
    Estimate how suitable a region is as a WB reference.

    Low Lab chroma and proximity to neutral a/b are stronger
    indicators than pixel count alone.
    """

    chroma_score = float(
        np.clip(
            1.0
            -
            chroma / 45.0,
            0.0,
            1.0,
        )
    )

    neutral_a = float(
        np.clip(
            1.0
            -
            abs(lab_a) / 18.0,
            0.0,
            1.0,
        )
    )

    neutral_b = float(
        np.clip(
            1.0
            -
            abs(lab_b) / 18.0,
            0.0,
            1.0,
        )
    )

    # Lab neutrality is more useful than raw chroma alone.
    neutrality = (
        chroma_score * 0.55
        +
        neutral_a * 0.225
        +
        neutral_b * 0.225
    )

    return float(
        np.clip(
            neutrality,
            0.0,
            1.0,
        )
    )


def estimate_white_balance(
    data: ImageColorData,
    mask: np.ndarray,
    *,
    max_pixels: int = 100_000,
) -> WhiteBalanceEstimate:
    """
    Estimate WB from a semantic/region mask.

    Important:
        This is an ESTIMATE, not a claim that the region is
        physically neutral.

    Strongly colored objects therefore receive low confidence.
    """

    image_shape = data.linear_rgb.shape[:2]

    mask = validate_mask_for_image(
        mask,
        image_shape,
    )

    rgb, weights = get_weighted_pixels(
        data.linear_rgb,
        mask,
        max_pixels=max_pixels,
        seed=71,
    )

    if rgb.shape[0] < WB_MIN_PIXELS:

        return WhiteBalanceEstimate(
            temperature=0.0,
            tint=0.0,

            red_factor=1.0,
            green_factor=1.0,
            blue_factor=1.0,

            lab_a=0.0,
            lab_b=0.0,

            chroma=0.0,
            neutrality=0.0,

            channel_agreement=0.0,

            confidence=0.0,
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    r = weighted_mean(
        rgb[:, 0],
        weights,
    )

    g = weighted_mean(
        rgb[:, 1],
        weights,
    )

    b = weighted_mean(
        rgb[:, 2],
        weights,
    )

    mean_rgb = (
        r + g + b
    ) / 3.0

    if mean_rgb <= EPS:

        return WhiteBalanceEstimate(
            temperature=0.0,
            tint=0.0,

            red_factor=1.0,
            green_factor=1.0,
            blue_factor=1.0,

            lab_a=0.0,
            lab_b=0.0,

            chroma=0.0,
            neutrality=0.0,

            channel_agreement=0.0,

            confidence=0.0,
        )

    # --------------------------------------------------------
    # CHANNEL BALANCE
    # --------------------------------------------------------

    blue_error, green_error = (
        _channel_balance_error(
            r,
            g,
            b,
        )
    )

    r_ratio = safe_ratio(
        r,
        mean_rgb,
        default=1.0,
    )

    g_ratio = safe_ratio(
        g,
        mean_rgb,
        default=1.0,
    )

    b_ratio = safe_ratio(
        b,
        mean_rgb,
        default=1.0,
    )

    # --------------------------------------------------------
    # LAB
    # --------------------------------------------------------

    lab, lab_weights = get_weighted_pixels(
        data.lab,
        mask,
        max_pixels=max_pixels,
        seed=73,
    )

    if lab.shape[0] < WB_MIN_PIXELS:

        return WhiteBalanceEstimate(
            temperature=0.0,
            tint=0.0,

            red_factor=1.0,
            green_factor=1.0,
            blue_factor=1.0,

            lab_a=0.0,
            lab_b=0.0,

            chroma=0.0,
            neutrality=0.0,

            channel_agreement=0.0,

            confidence=0.0,
        )

    mean_a = weighted_mean(
        lab[:, 1],
        lab_weights,
    )

    mean_b = weighted_mean(
        lab[:, 2],
        lab_weights,
    )

    chroma_values = np.sqrt(
        lab[:, 1] ** 2
        +
        lab[:, 2] ** 2
    )

    chroma = weighted_mean(
        chroma_values,
        lab_weights,
    )

    neutrality = _wb_neutrality(
        chroma,
        mean_a,
        mean_b,
    )

    # --------------------------------------------------------
    # RGB/LAB AGREEMENT
    # --------------------------------------------------------

    # RGB blue/yellow signal and Lab b should generally agree
    # in direction, although their scales are different.
    rgb_blue_direction = np.sign(
        blue_error
    )

    lab_blue_direction = np.sign(
        mean_b
    )

    if (
        abs(blue_error) < 0.003
        or
        abs(mean_b) < 2.0
    ):
        blue_agreement = 1.0
    elif (
        rgb_blue_direction
        ==
        lab_blue_direction
    ):
        blue_agreement = 1.0
    else:
        blue_agreement = 0.0

    rgb_green_direction = np.sign(
        green_error
    )

    lab_green_direction = np.sign(
        mean_a
    )

    if (
        abs(green_error) < 0.003
        or
        abs(mean_a) < 2.0
    ):
        green_agreement = 1.0
    elif (
        rgb_green_direction
        ==
        lab_green_direction
    ):
        green_agreement = 1.0
    else:
        green_agreement = 0.0

    channel_agreement = (
        blue_agreement
        +
        green_agreement
    ) * 0.5

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    # Positive Lab b generally means yellow.
    # A positive blue RGB imbalance means blue.
    #
    # The correction direction is therefore opposite to the
    # detected cast.
    rgb_temperature_signal = (
        -blue_error
        * 110.0
    )

    lab_temperature_signal = (
        -mean_b
        * 0.18
    )

    temperature = (
        rgb_temperature_signal * 0.55
        +
        lab_temperature_signal * 0.45
    )

    # --------------------------------------------------------
    # TINT
    # --------------------------------------------------------

    # Positive Lab a is magenta.
    # Positive green imbalance is green.
    #
    # Again, correction is opposite to the detected cast.
    rgb_tint_signal = (
        -green_error
        * 120.0
    )

    lab_tint_signal = (
        -mean_a
        * 0.20
    )

    tint = (
        rgb_tint_signal * 0.45
        +
        lab_tint_signal * 0.55
    )

    # Colored objects should not be aggressively neutralized.
    chroma_scale = (
        0.20
        +
        0.80 * neutrality
    )

    temperature *= chroma_scale
    tint *= chroma_scale

    # Reduce correction when RGB and Lab disagree.
    agreement_scale = (
        0.50
        +
        0.50 * channel_agreement
    )

    temperature *= agreement_scale
    tint *= agreement_scale

    temperature = clip(
        temperature,
        -TEMPERATURE_LIMIT,
        TEMPERATURE_LIMIT,
    )

    tint = clip(
        tint,
        -TINT_LIMIT,
        TINT_LIMIT,
    )

    # --------------------------------------------------------
    # CHANNEL GAINS
    # --------------------------------------------------------

    target = (
        r + g + b
    ) / 3.0

    red_factor = _safe_gain(
        target,
        r,
    )

    green_factor = _safe_gain(
        target,
        g,
    )

    blue_factor = _safe_gain(
        target,
        b,
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    effective_pixels = float(
        np.sum(weights)
    )

    pixel_confidence = float(
        np.clip(
            effective_pixels
            /
            5_000.0,
            0.0,
            1.0,
        )
    )

    # Very dark or very bright regions are poorer WB references.
    brightness_confidence = float(
        np.clip(
            (
                mean_rgb
                /
                0.12
            ),
            0.0,
            1.0,
        )
    )

    if mean_rgb > 0.85:
        brightness_confidence *= 0.65

    confidence = (
        pixel_confidence * 0.30
        +
        neutrality * 0.40
        +
        channel_agreement * 0.20
        +
        brightness_confidence * 0.10
    )

    confidence = float(
        np.clip(
            confidence,
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

        chroma=chroma,
        neutrality=neutrality,

        channel_agreement=channel_agreement,

        confidence=confidence,
    )


# ============================================================
# EXPOSURE
# ============================================================

def _scene_target_luminance(
    stats: RegionStats,
) -> float:
    """
    Estimate an appropriate median luminance target.

    0.18 is only the baseline. The target moves according to
    the tonal structure of the scene.
    """

    p10 = stats.p10
    p50 = stats.p50
    p90 = stats.p90

    # Start with photographic middle gray.
    target = 0.18

    # Very high-key regions should not be dragged down.
    if (
        p10 > 0.18
        and
        p50 > 0.35
        and
        p90 > 0.75
    ):
        target = 0.24

    # Very low-key regions should not automatically be lifted
    # toward middle gray.
    elif (
        p90 < 0.45
        and
        p50 < 0.14
    ):
        target = 0.14

    # Broadly balanced scenes remain near 0.18.
    return target


def exposure_from_luminance(
    stats: RegionStats,
) -> float:
    """
    Calculate exposure correction in EV.

    Uses:
        - median
        - lower percentile
        - upper percentile
        - clipping
        - scene-key adjustment

    rather than forcing every image to 18% luminance.
    """

    median = max(
        stats.p50,
        0.001,
    )

    target = _scene_target_luminance(
        stats
    )

    ev = math.log2(
        target
        /
        median
    )

    # --------------------------------------------------------
    # Protect highlights.
    # --------------------------------------------------------

    if stats.p95 > 0.92:
        excess = (
            stats.p95
            -
            0.92
        )

        ev *= max(
            0.35,
            1.0
            -
            excess * 5.0,
        )

    if stats.p99 > 0.98:
        ev *= 0.45

    # --------------------------------------------------------
    # Protect deep shadows.
    # --------------------------------------------------------

    if (
        stats.p10 < 0.015
        and
        stats.shadow_fraction > 0.15
    ):
        # Don't blindly raise a genuinely dark scene.
        ev *= 0.85

    # --------------------------------------------------------
    # Avoid huge corrections when distribution already spans
    # a healthy range.
    # --------------------------------------------------------

    dynamic_range = (
        stats.p90
        -
        stats.p10
    )

    if dynamic_range > 0.70:
        ev *= 0.80

    return clip(
        ev,
        EXPOSURE_MIN_EV,
        EXPOSURE_MAX_EV,
    )


def context_exposure_adjustment(
    region_stats: RegionStats,
    surrounding_stats: Optional[RegionStats],
) -> float:
    """Gently align a region with its immediate surroundings."""

    if surrounding_stats is None:
        return 0.0

    region_median = max(
        float(region_stats.p50),
        0.001,
    )

    surrounding_median = max(
        float(surrounding_stats.p50),
        0.001,
    )

    gap = math.log2(
        surrounding_median
        /
        region_median
    )

    # Match only part of the local contrast so intentional lighting survives.
    if abs(gap) < 0.08:
        return 0.0

    return clip(
        gap * 0.22,
        -0.35,
        0.35,
    )


def context_white_balance(
    region_wb: WhiteBalanceEstimate,
    surrounding_wb: Optional[WhiteBalanceEstimate],
    mask_name: str,
) -> Tuple[float, float]:
    """Return a restrained white-balance correction for a semantic region."""

    if surrounding_wb is None:
        return (
            clip(
                region_wb.temperature,
                -TEMPERATURE_LIMIT,
                TEMPERATURE_LIMIT,
            ),
            clip(
                region_wb.tint,
                -TINT_LIMIT,
                TINT_LIMIT,
            ),
        )

    region_confidence = float(
        np.clip(
            region_wb.confidence,
            0.0,
            1.0,
        )
    )

    context_confidence = float(
        np.clip(
            surrounding_wb.confidence,
            0.0,
            1.0,
        )
    )

    confidence = min(
        region_confidence,
        context_confidence,
    )

    temperature = (
        region_wb.temperature * 0.65
        +
        (
            region_wb.temperature
            -
            surrounding_wb.temperature
        )
        * 0.35
    )

    tint = (
        region_wb.tint * 0.65
        +
        (
            region_wb.tint
            -
            surrounding_wb.tint
        )
        * 0.35
    )

    return (
        clip(
            temperature * confidence,
            -TEMPERATURE_LIMIT,
            TEMPERATURE_LIMIT,
        ),
        clip(
            tint * confidence,
            -TINT_LIMIT,
            TINT_LIMIT,
        ),
    )


# ============================================================
# SHADOW / HIGHLIGHT ANALYSIS
# ============================================================

def calculate_tone_adjustments(
    stats: RegionStats,
) -> Dict[str, float]:
    """
    Estimate Lightroom-style tone controls.

    These are deliberately conservative because exposure,
    highlights, shadows, whites and blacks interact.
    """

    shadows = 0.0
    highlights = 0.0
    whites = 0.0
    blacks = 0.0

    # --------------------------------------------------------
    # SHADOWS
    # --------------------------------------------------------

    shadow_pressure = max(
        stats.shadow_fraction - 0.06,
        0.0,
    )

    if shadow_pressure > 0.0:

        shadows = clip(
            shadow_pressure * 180.0,
            0.0,
            35.0,
        )

    elif stats.p10 < 0.025:

        shadows = clip(
            (
                0.025
                -
                stats.p10
            )
            * 220.0,
            0.0,
            25.0,
        )

    # Don't lift shadows if the scene is intentionally low-key.
    if (
        stats.p50 < 0.12
        and
        stats.p90 < 0.45
    ):
        shadows *= 0.45

    # --------------------------------------------------------
    # HIGHLIGHTS
    # --------------------------------------------------------

    highlight_pressure = max(
        stats.highlight_fraction - 0.04,
        0.0,
    )

    if highlight_pressure > 0.0:

        highlights = -clip(
            highlight_pressure * 210.0,
            0.0,
            45.0,
        )

    elif stats.p95 > 0.94:

        highlights = -clip(
            (
                stats.p95
                -
                0.94
            )
            * 220.0,
            0.0,
            30.0,
        )

    # --------------------------------------------------------
    # WHITES
    # --------------------------------------------------------

    if stats.p99 > 0.985:

        whites = -clip(
            (
                stats.p99
                -
                0.985
            )
            * 220.0,
            0.0,
            25.0,
        )

    elif (
        stats.p95 < 0.65
        and
        stats.p99 < 0.82
    ):

        whites = clip(
            (
                0.65
                -
                stats.p95
            )
            * 35.0,
            0.0,
            15.0,
        )

    # --------------------------------------------------------
    # BLACKS
    # --------------------------------------------------------

    # Do NOT lift blacks simply because p01 is low.
    # Real photographs often contain legitimately black areas.

    if (
        stats.p01 > 0.025
        and
        stats.p10 > 0.12
    ):

        blacks = -clip(
            (
                stats.p01
                -
                0.025
            )
            * 80.0,
            0.0,
            10.0,
        )

    # Only lift blacks when the low end is compressed but there
    # is little genuine shadow information.
    elif (
        stats.p01 < 0.003
        and
        stats.p05 < 0.012
        and
        stats.shadow_fraction > 0.20
        and
        stats.p50 < 0.15
    ):

        blacks = clip(
            4.0,
            0.0,
            8.0,
        )

    return {
        "shadows": clip(
            shadows,
            0.0,
            40.0,
        ),

        "highlights": clip(
            highlights,
            -50.0,
            0.0,
        ),

        "whites": clip(
            whites,
            -30.0,
            20.0,
        ),

        "blacks": clip(
            blacks,
            -20.0,
            10.0,
        ),
    }


def calculate_dehaze(
    stats: RegionStats,
) -> float:
    """Estimate restrained dehaze from washed-out region statistics."""

    haze_pressure = max(
        0.0,
        0.48 - stats.rms_contrast,
    )

    highlight_penalty = max(
        0.0,
        stats.highlight_fraction - 0.08,
    )

    dehaze = (
        haze_pressure * 18.0
        -
        highlight_penalty * 12.0
    )

    return clip(
        dehaze,
        0.0,
        12.0,
    )


# ============================================================
# CONTRAST
# ============================================================

def calculate_contrast(
    stats: RegionStats,
) -> float:
    """
    Estimate global contrast correction.

    Uses percentile spread and clipping rather than a universal
    target dynamic range.
    """

    dynamic_range = (
        stats.p90
        -
        stats.p10
    )

    # Normal photographic range.
    low_target = 0.42
    high_target = 0.68

    correction = 0.0

    if dynamic_range < low_target:

        correction = (
            low_target
            -
            dynamic_range
        ) * 65.0

    elif dynamic_range > high_target:

        correction = -(
            dynamic_range
            -
            high_target
        ) * 45.0

    # Clipping should strongly suppress contrast increases.
    if (
        stats.highlight_fraction > 0.06
        or
        stats.shadow_fraction > 0.12
    ):
        correction *= 0.55

    return clip(
        correction,
        -20.0,
        20.0,
    )


# ============================================================
# COLOR INTENSITY
# ============================================================

def _estimate_saturation(
    data: ImageColorData,
    mask: np.ndarray,
    max_pixels: int = 100_000,
) -> float:
    """
    Obtain saturation without requiring HSV.

    If HSV is available, use it.
    Otherwise use normalized Lab chroma.
    """

    if data.hsv is not None:

        hsv, weights = get_weighted_pixels(
            data.hsv,
            mask,
            max_pixels=max_pixels,
            seed=81,
        )

        if hsv.size == 0:
            return 0.0

        return weighted_mean(
            hsv[:, 1],
            weights,
        )

    lab, weights = get_weighted_pixels(
        data.lab,
        mask,
        max_pixels=max_pixels,
        seed=83,
    )

    if lab.size == 0:
        return 0.0

    chroma = np.sqrt(
        lab[:, 1] ** 2
        +
        lab[:, 2] ** 2
    )

    return float(
        np.clip(
            weighted_mean(
                chroma,
                weights,
            )
            /
            100.0,
            0.0,
            1.0,
        )
    )


def calculate_color_adjustments(
    data: ImageColorData,
    mask: np.ndarray,
    mask_name: str,
) -> Dict[str, float]:
    """
    Calculate conservative saturation/vibrance adjustments.
    """

    saturation = _estimate_saturation(
        data,
        mask,
    )

    saturation_adjustment = 0.0

    if saturation < 0.12:

        saturation_adjustment = clip(
            (
                0.12
                -
                saturation
            )
            * 35.0,
            0.0,
            7.0,
        )

    elif saturation > 0.70:

        saturation_adjustment = -clip(
            (
                saturation
                -
                0.70
            )
            * 35.0,
            0.0,
            12.0,
        )

    # Vibrance is intentionally smaller.
    if saturation < 0.35:

        vibrance = clip(
            (
                0.35
                -
                saturation
            )
            * 20.0,
            0.0,
            8.0,
        )

    else:

        vibrance = 0.0

    # --------------------------------------------------------
    # Semantic protection.
    # --------------------------------------------------------

    skin_masks = {
        "skin",
        "skin_face",
        "skin_neck",
        "face",
        "hands",
        "arms",
    }

    if mask_name in skin_masks:

        saturation_adjustment *= 0.30
        vibrance *= 0.20

    elif mask_name in {
        "eyes",
        "lips",
        "hair",
    }:

        saturation_adjustment *= 0.55
        vibrance *= 0.50

    return {
        "saturation": clip(
            saturation_adjustment,
            -20.0,
            20.0,
        ),

        "vibrance": clip(
            vibrance,
            -15.0,
            15.0,
        ),
    }


# ============================================================
# MICRO-CONTRAST / TEXTURE
# ============================================================

def local_detail_metric(
    luminance: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Estimate local high-frequency luminance detail.

    This is a texture proxy, not a full clarity estimator.
    """

    if luminance.ndim != 2:
        raise ValueError(
            "luminance must be 2D."
        )

    mask = validate_mask_for_image(
        mask,
        luminance.shape,
    )

    if np.count_nonzero(
        mask > 0.15
    ) < 20:
        return 0.0

    blurred = cv2.GaussianBlur(
        luminance.astype(
            np.float32,
            copy=False,
        ),
        (0, 0),
        2.0,
    )

    detail = np.abs(
        luminance
        -
        blurred
    )

    values, weights = get_weighted_pixels(
        detail,
        mask,
        max_pixels=100_000,
        seed=91,
    )

    if values.size == 0:
        return 0.0

    return weighted_percentile(
        values,
        weights,
        50.0,
    )


def calculate_clarity(
    data: ImageColorData,
    mask: np.ndarray,
    mask_name: str,
) -> float:
    """
    Estimate a conservative local-clarity value for a region mask.

    This is intentionally a lightweight proxy. It combines local edge detail
    with a subtle texture/contrast contribution while protecting sensitive
    semantic masks such as skin and lips from exaggerated sharpness.
    """

    if data is None or data.luminance is None:
        return 0.0

    mask = validate_mask_for_image(
        mask,
        data.luminance.shape,
    )

    if np.count_nonzero(
        mask > 0.15
    ) < 20:
        return 0.0

    detail = local_detail_metric(
        data.luminance,
        mask,
    )

    stats = calculate_region_stats(
        data,
        mask,
    )

    clarity = detail * 1.75

    if stats is not None:
        local_contrast = float(
            np.clip(
                stats.rms_contrast * 3.0,
                0.0,
                0.40,
            )
        )

        dynamic_range = float(
            np.clip(
                (stats.p90 - stats.p10) * 2.0,
                0.0,
                0.50,
            )
        )

        clarity += local_contrast * 0.55
        clarity += dynamic_range * 0.35

    # Keep the signal conservative and avoid aggressive sharpening on
    # sensitive region types.
    if mask_name in {
        "skin",
        "skin_face",
        "skin_neck",
        "face",
    }:
        clarity *= 0.55

    elif mask_name in {
        "eyes",
        "hair",
    }:
        clarity *= 1.15

    elif mask_name in {
        "lips",
        "mouth",
    }:
        clarity *= 0.70

    return float(
        np.clip(
            clarity,
            0.0,
            0.35,
        )
    )