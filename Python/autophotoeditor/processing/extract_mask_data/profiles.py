from __future__ import annotations

from .shared import *
from .masks import *
from .analysis import *

@dataclass
class EnhancementValues:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0

    temperature: float = 0.0
    tint: float = 0.0

    saturation: float = 0.0
    vibrance: float = 0.0

    clarity: float = 0.0
    dehaze: float = 0.0


def limit_values(
    values: EnhancementValues,
) -> EnhancementValues:

    result = {}

    for field_name in values.__dataclass_fields__:

        value = getattr(
            values,
            field_name,
        )

        low, high = LIMITS[
            field_name
        ]

        result[field_name] = clip(
            value,
            low,
            high,
        )

    return EnhancementValues(
        **result
    )


def apply_strength(
    values: EnhancementValues,
    strength: float,
) -> EnhancementValues:

    result = {}

    for field_name in values.__dataclass_fields__:

        value = getattr(
            values,
            field_name,
        )

        result[field_name] = (
            value
            *
            strength
        )

    return EnhancementValues(
        **result
    )


# ============================================================
# SPECIAL MASK LOGIC
# ============================================================

def adjust_for_mask_type(
    values: EnhancementValues,
    name: str,
) -> EnhancementValues:

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if name == "face":

        values.contrast *= 0.55
        values.clarity *= 0.25
        values.dehaze *= 0.25
        values.saturation *= 0.45

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    elif name in {
        "skin",
        "skin_face",
        "skin_neck",
    }:

        values.contrast *= 0.30
        values.clarity *= 0.08
        values.dehaze *= 0.10

        values.saturation *= 0.25
        values.vibrance *= 0.20

        values.highlights *= 0.70
        values.shadows *= 0.80

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    elif name == "eyes":

        values.exposure *= 0.40
        values.contrast *= 0.50
        values.clarity *= 0.80
        values.dehaze *= 0.25

    # --------------------------------------------------------
    # LIPS
    # --------------------------------------------------------

    elif name in {
        "lips",
        "mouth",
    }:

        values.contrast *= 0.45
        values.clarity *= 0.35
        values.saturation *= 0.35

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    elif name == "hair":

        values.contrast *= 0.65
        values.clarity *= 0.65
        values.dehaze *= 0.40

    # --------------------------------------------------------
    # CLOTHING
    # --------------------------------------------------------

    elif name in {
        "clothing",
        "top",
        "dress",
        "skirt",
        "pants",
        "torso",
    }:

        values.contrast *= 0.75
        values.clarity *= 0.75

    # --------------------------------------------------------
    # SMALL DETAILS
    # --------------------------------------------------------

    elif name in {
        "jewelry",
        "glasses",
        "brows",
    }:

        values.exposure *= 0.40
        values.contrast *= 0.45
        values.clarity *= 0.50

    return values


# ============================================================
# AUTO MASK ANALYZER
# ============================================================

class AutoMaskEnhancer:

    def __init__(
        self,
        image: np.ndarray,
        hard_masks: Dict[str, np.ndarray],
        soft_masks: Dict[str, np.ndarray],
        global_strength: float = 1.0,
    ):

        self.image = image

        self.hard_masks = hard_masks
        self.soft_masks = soft_masks

        self.global_strength = clip(
            global_strength,
            0.0,
            2.0,
        )

        logger.info(
            "Preparing color-science image data..."
        )

        self.data = prepare_color_data(
            image
        )

        self.global_stats = (
            calculate_global_stats(
                self.data
            )
        )

        logger.info(
            "Global luminance median: %.4f",
            self.global_stats.p50,
        )

        logger.info(
            "Global luminance range: %.4f - %.4f",
            self.global_stats.p05,
            self.global_stats.p95,
        )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    def get_mask(
        self,
        name: str,
    ) -> Optional[np.ndarray]:

        hard = self.hard_masks.get(
            name
        )

        soft = self.soft_masks.get(
            name
        )

        if soft is not None:

            # Combine hard segmentation and soft confidence.
            if hard is not None:

                return (
                    soft
                    *
                    (
                        hard > 0
                    ).astype(
                        np.float32
                    )
                )

            return soft

        if hard is not None:
            return hard.astype(
                np.float32
            )

        return None

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def analyze_mask(
        self,
        name: str,
    ) -> Optional[Dict[str, Any]]:

        mask = self.get_mask(
            name
        )

        if mask is None:
            return None

        pixel_count = int(
            np.count_nonzero(
                mask > 0.15
            )
        )

        if pixel_count == 0:
            return None

        minimum = MIN_PIXELS.get(
            name,
            50,
        )

        if pixel_count < minimum:

            logger.debug(
                "Mask %s is too small: %d pixels",
                name,
                pixel_count,
            )

            return {
                "enabled": False,
                "reason": "mask_too_small",
                "pixel_count": pixel_count,
                "adjustments": asdict(
                    EnhancementValues()
                ),
            }

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        region_stats = calculate_region_stats(
            self.data,
            mask,
        )

        if region_stats is None:
            return None

        # ----------------------------------------------------
        # Surrounding region
        # ----------------------------------------------------

        ring = surrounding_ring(
            mask,
            self.image.shape[:2],
        )

        surrounding_stats = (
            calculate_region_stats(
                self.data,
                ring.astype(
                    np.float32
                ),
            )
        )

        # ----------------------------------------------------
        # White balance
        # ----------------------------------------------------

        region_wb = estimate_white_balance(
            self.data,
            mask,
        )

        surrounding_wb = None

        if surrounding_stats is not None:
            surrounding_wb = (
                estimate_white_balance(
                    self.data,
                    ring.astype(
                        np.float32
                    ),
                )
            )

        # ----------------------------------------------------
        # Exposure
        # ----------------------------------------------------

        exposure = exposure_from_luminance(
            region_stats
        )

        # Context correction.
        exposure += context_exposure_adjustment(
            region_stats,
            surrounding_stats,
        )

        # ----------------------------------------------------
        # Tone
        # ----------------------------------------------------

        tone = calculate_tone_adjustments(
            region_stats
        )

        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        contrast = calculate_contrast(
            region_stats
        )

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = calculate_color_adjustments(
            region_stats,
            name,
        )

        # ----------------------------------------------------
        # Clarity
        # ----------------------------------------------------

        clarity = calculate_clarity(
            self.data,
            mask,
            name,
        )

        # ----------------------------------------------------
        # Dehaze
        # ----------------------------------------------------

        dehaze = calculate_dehaze(
            region_stats
        )

        # ----------------------------------------------------
        # White balance
        # ----------------------------------------------------

        temperature, tint = (
            context_white_balance(
                region_wb,
                surrounding_wb,
                name,
            )
        )

        # ----------------------------------------------------
        # Assemble
        # ----------------------------------------------------

        values = EnhancementValues(
            exposure=exposure,

            contrast=contrast,

            highlights=tone[
                "highlights"
            ],

            shadows=tone[
                "shadows"
            ],

            whites=tone[
                "whites"
            ],

            blacks=tone[
                "blacks"
            ],

            temperature=temperature,
            tint=tint,

            saturation=color[
                "saturation"
            ],

            vibrance=color[
                "vibrance"
            ],

            clarity=clarity,
            dehaze=dehaze,
        )

        # ----------------------------------------------------
        # Mask-specific safety rules
        # ----------------------------------------------------

        values = adjust_for_mask_type(
            values,
            name,
        )

        # ----------------------------------------------------
        # Per-mask strength
        # ----------------------------------------------------

        mask_strength = MASK_STRENGTH.get(
            name,
            0.40,
        )

        values = apply_strength(
            values,
            mask_strength
            *
            self.global_strength,
        )

        context_match = 0.5
        context_luminance_gap = 0.0
        context_chroma_gap = 0.0
        if surrounding_stats is not None:
            context_luminance_gap = abs(
                math.log2(
                    max(region_stats.luminance_median, EPS)
                    / max(surrounding_stats.luminance_median, EPS)
                )
            )
            context_chroma_gap = math.sqrt(
                (region_stats.lab_a - surrounding_stats.lab_a) ** 2
                + (region_stats.lab_b - surrounding_stats.lab_b) ** 2
            ) / 40.0
            context_match = float(np.clip(
                1.0
                - 0.55 * min(context_luminance_gap, 1.5)
                - 0.45 * min(context_chroma_gap, 1.5),
                0.0,
                1.0,
            ))

        # Reduce the recommendation when its source region is a poor match
        # for the local surroundings. This keeps a strong segmentation from
        # becoming a visible color or luminance island.
        safety_scale = float(np.clip(0.45 + 0.55 * context_match, 0.35, 1.0))
        values = apply_strength(values, safety_scale)

        values = limit_values(
            values
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        coverage = (
            pixel_count
            /
            max(
                self.image.shape[0]
                *
                self.image.shape[1],
                1,
            )
        )

        confidence = float(
            np.clip(
                math.log10(
                    pixel_count + 10
                )
                /
                5.0,
                0.0,
                1.0,
            )
        )

        if surrounding_stats is not None:
            confidence = min(
                1.0,
                confidence + 0.15,
            )

        return {
            "enabled": True,

            "pixel_count": pixel_count,

            "coverage": coverage,

            "confidence": confidence,

            "context_match": {
                "score": context_match,
                "luminance_gap_ev": context_luminance_gap,
                "chroma_gap": context_chroma_gap,
                "safety_scale": safety_scale,
            },

            "analysis": {
                "luminance": {
                    "mean": region_stats.luminance_mean,
                    "median": region_stats.luminance_median,
                    "p01": region_stats.p01,
                    "p05": region_stats.p05,
                    "p10": region_stats.p10,
                    "p25": region_stats.p25,
                    "p50": region_stats.p50,
                    "p75": region_stats.p75,
                    "p90": region_stats.p90,
                    "p95": region_stats.p95,
                    "p99": region_stats.p99,
                },

                "contrast": {
                    "rms": region_stats.rms_contrast,
                    "dynamic_range": (
                        region_stats.p95
                        -
                        region_stats.p05
                    ),
                },

                "clipping": {
                    "shadow_fraction": (
                        region_stats.shadow_fraction
                    ),
                    "highlight_fraction": (
                        region_stats.highlight_fraction
                    ),
                },

                "color": {
                    "lab_L": region_stats.lab_L,
                    "lab_a": region_stats.lab_a,
                    "lab_b": region_stats.lab_b,
                    "chroma": region_stats.chroma,
                    "red_green_ratio": (
                        region_stats.red_green_ratio
                    ),
                    "blue_yellow_ratio": (
                        region_stats.blue_yellow_ratio
                    ),
                    "saturation": region_stats.saturation,
                },

                "white_balance": {
                    "temperature_estimate": (
                        region_wb.temperature
                    ),
                    "tint_estimate": (
                        region_wb.tint
                    ),
                    "confidence": (
                        region_wb.confidence
                    ),
                },

                "surrounding": (
                    {
                        "luminance_median": (
                            surrounding_stats.luminance_median
                        ),
                        "luminance_mean": (
                            surrounding_stats.luminance_mean
                        ),
                        "lab_a": (
                            surrounding_stats.lab_a
                        ),
                        "lab_b": (
                            surrounding_stats.lab_b
                        ),
                    }
                    if surrounding_stats is not None
                    else None
                ),
            },

            "adjustments": asdict(
                values
            ),
        }

    # --------------------------------------------------------
    # ALL MASKS
    # --------------------------------------------------------

    def analyze(
        self,
    ) -> Dict[str, Any]:

        results = {}

        # Avoid processing aliases twice unnecessarily.
        for name in self.hard_masks:

            if name in {
                "background",
                "non_person",
            }:
                continue

            logger.info(
                "Analyzing mask: %s",
                name,
            )

            result = self.analyze_mask(
                name
            )

            if result is not None:
                results[name] = result

        return results


