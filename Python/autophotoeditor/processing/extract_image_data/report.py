from __future__ import annotations

from .shared import *
from .tone import *
from .white_balance import *


# ============================================================================
# TONE STATS
# ============================================================================

@dataclass(frozen=True)
class ToneStats:
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
    std: float
    shadow_clip: float
    highlight_clip: float


def calculate_tone_stats(
    values: np.ndarray,
) -> ToneStats:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return ToneStats(
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
            std=0.0,
            shadow_clip=0.0,
            highlight_clip=0.0,
        )

    arr = np.clip(arr, 0.0, 1.0)

    p01 = percentile(arr, 1)
    p05 = percentile(arr, 5)
    p10 = percentile(arr, 10)
    p25 = percentile(arr, 25)
    p50 = percentile(arr, 50)
    p75 = percentile(arr, 75)
    p90 = percentile(arr, 90)
    p95 = percentile(arr, 95)
    p99 = percentile(arr, 99)

    return ToneStats(
        p01=safe_float(p01, 0.0),
        p05=safe_float(p05, 0.0),
        p10=safe_float(p10, 0.0),
        p25=safe_float(p25, 0.0),
        p50=safe_float(p50, 0.0),
        p75=safe_float(p75, 0.0),
        p90=safe_float(p90, 0.0),
        p95=safe_float(p95, 0.0),
        p99=safe_float(p99, 0.0),
        mean=safe_float(np.mean(arr), 0.0),
        std=safe_float(np.std(arr), 0.0),
        shadow_clip=safe_float(np.mean(arr <= 0.003), 0.0),
        highlight_clip=safe_float(np.mean(arr >= 0.995), 0.0),
    )


# ============================================================================
# LOCAL HELPERS
# ============================================================================

def _safe_dict_value(
    data: Optional[Dict[str, Any]],
    key: str,
    default: Any,
) -> Any:
    if not isinstance(data, dict):
        return default

    value = data.get(key, default)

    if value is None:
        return default

    return value


def _safe_float_value(
    data: Optional[Dict[str, Any]],
    key: str,
    default: float = 0.0,
) -> float:
    return safe_float(
        _safe_dict_value(
            data,
            key,
            default,
        ),
        default,
    )


def _finite_array(
    values: np.ndarray,
) -> np.ndarray:

    arr = np.asarray(
        values
    )

    if arr.size == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    arr = arr.astype(
        np.float32,
        copy=False,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    return arr


def _safe_percentile(
    values: np.ndarray,
    q: float,
    default: float = 0.0,
) -> float:

    values = _finite_array(
        values
    )

    if values.size == 0:
        return default

    return safe_float(
        percentile(
            values,
            q,
        ),
        default,
    )


def _safe_mean(
    values: np.ndarray,
    default: float = 0.0,
) -> float:

    values = _finite_array(
        values
    )

    if values.size == 0:
        return default

    return safe_float(
        np.mean(values),
        default,
    )


def _prepare_mask(
    mask: Optional[np.ndarray],
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:

    if mask is None:
        return None

    m = np.asarray(
        mask
    )

    if m.size == 0:
        return None

    if m.ndim > 2:
        m = np.squeeze(m)

    if m.ndim != 2:
        return None

    target_h, target_w = target_shape

    if (
        m.shape[0] != target_h
        or m.shape[1] != target_w
    ):
        m = cv2.resize(
            m.astype(
                np.float32,
                copy=False,
            ),
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    m = normalized_mask(
        m
    )

    if m is None:
        return None

    return np.asarray(
        m,
        dtype=bool,
    )


def _sample_masked_values(
    values: np.ndarray,
    mask: np.ndarray,
    max_samples: int,
) -> np.ndarray:

    selected = values[
        mask
    ]

    if selected.size == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    return sample_array(
        selected,
        max_samples,
    )


def _safe_normalized_confidence(
    value: Any,
) -> float:

    return clamp(
        safe_float(
            value,
            0.0,
        ),
        0.0,
        1.0,
    )


def _wb_strength_from_confidence(
    confidence: float,
) -> float:
    """
    Smooth WB confidence curve.

    Very weak WB evidence:
        no correction

    Moderate evidence:
        restrained correction

    Strong evidence:
        full correction
    """

    confidence = clamp(
        confidence,
        0.0,
        1.0,
    )

    if confidence <= 0.15:
        return 0.0

    if confidence <= 0.35:
        t = (
            confidence - 0.15
        ) / 0.20

        # Smoothstep
        t = (
            t * t
            * (3.0 - 2.0 * t)
        )

        return 0.45 * t

    if confidence <= 0.70:
        t = (
            confidence - 0.35
        ) / 0.35

        t = (
            t * t
            * (3.0 - 2.0 * t)
        )

        return (
            0.45
            + 0.35 * t
        )

    t = (
        confidence - 0.70
    ) / 0.30

    t = (
        t * t
        * (3.0 - 2.0 * t)
    )

    return (
        0.80
        + 0.20 * t
    )


def _exposure_strength_from_confidence(
    confidence: float,
) -> float:

    confidence = clamp(
        confidence,
        0.0,
        1.0,
    )

    if confidence <= 0.20:
        return 0.0

    if confidence >= 0.75:
        return 1.0

    t = (
        confidence - 0.20
    ) / 0.55

    # Smooth transition.
    return (
        t * t
        * (3.0 - 2.0 * t)
    )


def _estimate_subject_luminance(
    y_linear: np.ndarray,
    subject_mask: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:

    if subject_mask is None:
        return None

    mask = _prepare_mask(
        subject_mask,
        y_linear.shape[:2],
    )

    if mask is None:
        return None

    count = int(
        np.count_nonzero(mask)
    )

    if count < 100:
        return None

    values = _sample_masked_values(
        y_linear,
        mask,
        50_000,
    )

    if values.size == 0:
        return None

    return {
        "coverage": round(
            float(
                count / mask.size
            ),
            6,
        ),
        "p10": round(
            _safe_percentile(
                values,
                10,
            ),
            6,
        ),
        "p25": round(
            _safe_percentile(
                values,
                25,
            ),
            6,
        ),
        "p50": round(
            _safe_percentile(
                values,
                50,
            ),
            6,
        ),
        "p75": round(
            _safe_percentile(
                values,
                75,
            ),
            6,
        ),
        "p90": round(
            _safe_percentile(
                values,
                90,
            ),
            6,
        ),
        "mean": round(
            _safe_mean(
                values
            ),
            6,
        ),
    }


# ============================================================================
# REGION ANALYSIS
# ============================================================================

def analyze_region(
    name: str,
    mask: Optional[np.ndarray],
    l_star: np.ndarray,
    rgb: np.ndarray,
) -> Optional[Dict[str, Any]]:

    if mask is None:
        return None

    if l_star.ndim != 2:
        return None

    if rgb.ndim != 3:
        return None

    if rgb.shape[:2] != l_star.shape[:2]:
        return None

    m = _prepare_mask(
        mask,
        l_star.shape[:2],
    )

    if m is None:
        return None

    count = int(
        np.count_nonzero(m)
    )

    if count < 20:
        return None

    l_values = _sample_masked_values(
        l_star,
        m,
        40_000,
    )

    rgb_selected = rgb[m]

    if rgb_selected.size == 0:
        return None

    rgb_values = sample_array(
        rgb_selected,
        40_000,
    )

    rgb_values = np.asarray(
        rgb_values,
        dtype=np.float32,
    )

    if rgb_values.ndim == 1:
        if rgb_values.size % 3 != 0:
            return None

        rgb_values = rgb_values.reshape(
            -1,
            3,
        )

    if rgb_values.shape[-1] != 3:
        return None

    return {
        "name": name,

        "coverage": round(
            float(
                count / m.size
            ),
            6,
        ),

        "luminance_l_star": {
            "p10": round(
                _safe_percentile(
                    l_values,
                    10,
                ),
                3,
            ),
            "p25": round(
                _safe_percentile(
                    l_values,
                    25,
                ),
                3,
            ),
            "p50": round(
                _safe_percentile(
                    l_values,
                    50,
                ),
                3,
            ),
            "p75": round(
                _safe_percentile(
                    l_values,
                    75,
                ),
                3,
            ),
            "p90": round(
                _safe_percentile(
                    l_values,
                    90,
                ),
                3,
            ),
        },

        "rgb_mean": [
            round(
                safe_float(
                    x,
                    0.0,
                ),
                5,
            )
            for x in np.mean(
                rgb_values,
                axis=0,
            )
        ],

        "rgb_std": [
            round(
                safe_float(
                    x,
                    0.0,
                ),
                5,
            )
            for x in np.std(
                rgb_values,
                axis=0,
            )
        ],
    }


# ============================================================================
# TONE PROFILE
# ============================================================================

def analyze_tone(
    y_linear: np.ndarray,
    l_star: np.ndarray,
) -> Dict[str, Any]:

    if y_linear.ndim != 2:
        raise ValueError(
            "y_linear must be a 2D luminance array."
        )

    if l_star.ndim != 2:
        raise ValueError(
            "l_star must be a 2D array."
        )

    if y_linear.shape != l_star.shape:
        raise ValueError(
            "y_linear and l_star must have identical dimensions."
        )

    y_values = sample_array(
        y_linear,
        MAX_OPTIMIZATION_PIXELS,
    )

    l_values = sample_array(
        l_star,
        MAX_OPTIMIZATION_PIXELS,
    )

    y_values = _finite_array(
        y_values
    )

    l_values = _finite_array(
        l_values
    )

    if y_values.size == 0:
        y_values = np.array(
            [0.0],
            dtype=np.float32,
        )

    if l_values.size == 0:
        l_values = np.array(
            [0.0],
            dtype=np.float32,
        )

    y_values = np.clip(
        y_values,
        0.0,
        1.0,
    )

    l_values = np.clip(
        l_values,
        0.0,
        100.0,
    )

    y_stats = calculate_tone_stats(
        y_values
    )

    l_normalized = np.clip(
        l_values / 100.0,
        0.0,
        1.0,
    )

    l_stats = calculate_tone_stats(
        l_normalized
    )

    y_p01 = max(
        y_stats.p01,
        EPS,
    )

    dynamic_range = (
        y_stats.p99
        / y_p01
    )

    # Ratio can become astronomically large on images with
    # almost-black pixels. Keep it useful and numerically stable.
    dynamic_range = clamp(
        dynamic_range,
        1.0,
        1_000_000.0,
    )

    return {
        "linear_luminance": {
            "p01": round(
                y_stats.p01,
                6,
            ),
            "p05": round(
                y_stats.p05,
                6,
            ),
            "p10": round(
                y_stats.p10,
                6,
            ),
            "p25": round(
                y_stats.p25,
                6,
            ),
            "p50": round(
                y_stats.p50,
                6,
            ),
            "p75": round(
                y_stats.p75,
                6,
            ),
            "p90": round(
                y_stats.p90,
                6,
            ),
            "p95": round(
                y_stats.p95,
                6,
            ),
            "p99": round(
                y_stats.p99,
                6,
            ),
            "mean": round(
                y_stats.mean,
                6,
            ),
            "std": round(
                y_stats.std,
                6,
            ),
        },

        "l_star": {
            "p01": round(
                _safe_percentile(
                    l_values,
                    1,
                ),
                3,
            ),
            "p05": round(
                _safe_percentile(
                    l_values,
                    5,
                ),
                3,
            ),
            "p10": round(
                _safe_percentile(
                    l_values,
                    10,
                ),
                3,
            ),
            "p25": round(
                _safe_percentile(
                    l_values,
                    25,
                ),
                3,
            ),
            "p50": round(
                _safe_percentile(
                    l_values,
                    50,
                ),
                3,
            ),
            "p75": round(
                _safe_percentile(
                    l_values,
                    75,
                ),
                3,
            ),
            "p90": round(
                _safe_percentile(
                    l_values,
                    90,
                ),
                3,
            ),
            "p95": round(
                _safe_percentile(
                    l_values,
                    95,
                ),
                3,
            ),
            "p99": round(
                _safe_percentile(
                    l_values,
                    99,
                ),
                3,
            ),
        },

        "clipping": {
            "shadow_fraction": round(
                clamp(
                    y_stats.shadow_clip,
                    0.0,
                    1.0,
                ),
                7,
            ),
            "highlight_fraction": round(
                clamp(
                    y_stats.highlight_clip,
                    0.0,
                    1.0,
                ),
                7,
            ),
        },

        "dynamic_range": round(
            dynamic_range,
            4,
        ),

        "tonal_spread": round(
            max(
                0.0,
                y_stats.p90
                - y_stats.p10,
            ),
            6,
        ),

        "midtone_separation": round(
            max(
                0.0,
                y_stats.p75
                - y_stats.p25,
            ),
            6,
        ),
    }


# ============================================================================
# LIGHTING QUALITY
# ============================================================================

def analyze_lighting_quality(
    y_linear: np.ndarray,
    scene_type: str,
    subject_mask: Optional[np.ndarray],
) -> Dict[str, Any]:

    if y_linear.ndim != 2:
        raise ValueError(
            "y_linear must be a 2D luminance array."
        )

    values = sample_array(
        y_linear,
        MAX_OPTIMIZATION_PIXELS,
    )

    values = _finite_array(
        values
    )

    values = np.clip(
        values,
        0.0,
        1.0,
    )

    if values.size == 0:
        values = np.array(
            [0.0],
            dtype=np.float32,
        )

    stats = calculate_tone_stats(
        values
    )

    shadow = clamp(
        stats.shadow_clip,
        0.0,
        1.0,
    )

    highlight = clamp(
        stats.highlight_clip,
        0.0,
        1.0,
    )

    tonal_spread = max(
        0.0,
        stats.p90
        - stats.p10,
    )

    midtone_spread = max(
        0.0,
        stats.p75
        - stats.p25,
    )

    # ------------------------------------------------------------------------
    # Subject analysis
    # ------------------------------------------------------------------------

    subject = _estimate_subject_luminance(
        y_linear,
        subject_mask,
    )

    subject_issues = []

    if subject is not None:

        subject_p10 = subject[
            "p10"
        ]

        subject_p50 = subject[
            "p50"
        ]

        subject_p90 = subject[
            "p90"
        ]

        subject_spread = (
            subject_p90
            - subject_p10
        )

        if subject_p50 < 0.13:
            subject_issues.append(
                "subject_dark"
            )

        elif subject_p50 > 0.78:
            subject_issues.append(
                "subject_bright"
            )

        if subject_spread < 0.12:
            subject_issues.append(
                "subject_low_tonal_separation"
            )

    # ------------------------------------------------------------------------
    # Scene-aware diagnostic thresholds
    # ------------------------------------------------------------------------

    scene = (
        str(scene_type or "")
        .strip()
        .lower()
    )

    # Conservative defaults.
    highlight_threshold = 0.012
    shadow_threshold = 0.035
    low_spread_threshold = 0.18
    dark_median_threshold = 0.19
    bright_median_threshold = 0.76

    # These are diagnostic thresholds only.
    # They do not directly modify the image.
    if scene in {
        "night",
        "low_light",
        "dark",
    }:
        shadow_threshold = 0.08
        dark_median_threshold = 0.12

    elif scene in {
        "snow",
        "beach",
        "high_key",
        "bright",
    }:
        bright_median_threshold = 0.84
        highlight_threshold = 0.025

    elif scene in {
        "portrait",
        "people",
    }:
        # Portraits commonly contain deliberately dark
        # backgrounds, so do not over-penalize shadows.
        shadow_threshold = 0.055

    # ------------------------------------------------------------------------
    # Global issues
    # ------------------------------------------------------------------------

    issues: List[str] = []

    if highlight > highlight_threshold:
        issues.append(
            "highlight_clipping"
        )

    if shadow > shadow_threshold:
        issues.append(
            "blocked_shadows"
        )

    if tonal_spread < low_spread_threshold:
        issues.append(
            "low_tonal_separation"
        )

    if midtone_spread < 0.10:
        issues.append(
            "weak_midtone_separation"
        )

    if stats.p50 < dark_median_threshold:
        issues.append(
            "globally_dark"
        )

    if stats.p50 > bright_median_threshold:
        issues.append(
            "globally_bright"
        )

    # Subject-specific diagnostics are more useful than
    # blindly treating the entire image as one exposure region.
    for issue in subject_issues:
        if issue not in issues:
            issues.append(
                issue
            )

    if not issues:
        lighting_state = "balanced"

    elif len(issues) == 1:
        lighting_state = issues[0]

    else:
        lighting_state = "mixed"

    result = {
        "state": lighting_state,

        "issues": issues,

        "median_linear": round(
            stats.p50,
            6,
        ),

        "tonal_spread": round(
            tonal_spread,
            6,
        ),

        "midtone_spread": round(
            midtone_spread,
            6,
        ),

        "shadow_clipping": round(
            shadow,
            7,
        ),

        "highlight_clipping": round(
            highlight,
            7,
        ),

        "scene_type": scene_type,
    }

    if subject is not None:
        result[
            "subject"
        ] = subject

    return result


# ============================================================================
# WHITE-BALANCE SAFETY
# ============================================================================

def _get_safe_wb_gains(
    wb: Dict[str, Any],
) -> np.ndarray:

    raw = wb.get(
        "gains_rgb",
        [1.0, 1.0, 1.0],
    )

    try:
        gains = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(-1)
    except Exception:
        gains = np.ones(
            3,
            dtype=np.float32,
        )

    if gains.size != 3:
        gains = np.ones(
            3,
            dtype=np.float32,
        )

    gains = np.where(
        np.isfinite(gains),
        gains,
        1.0,
    )

    gains = np.clip(
        gains,
        MIN_WB_GAIN,
        MAX_WB_GAIN,
    )

    return normalize_rgb_gains(
        gains
    )


def _get_wb_reliability(
    wb: Dict[str, Any],
) -> float:

    confidence = _safe_normalized_confidence(
        wb.get(
            "confidence",
            0.0,
        )
    )

    # If the WB module reports estimator agreement,
    # use it as an additional safety signal.
    agreement = wb.get(
        "agreement",
        None,
    )

    if agreement is None:
        agreement = wb.get(
            "estimator_agreement",
            None,
        )

    if agreement is None:
        return confidence

    agreement = _safe_normalized_confidence(
        agreement
    )

    # Confidence remains dominant, but disagreement
    # prevents aggressive correction.
    return clamp(
        0.70 * confidence
        + 0.30 * agreement,
        0.0,
        1.0,
    )


# ============================================================================
# COMPATIBILITY RECOMMENDATIONS
# ============================================================================

def derive_edit_recommendations(
    exposure: Dict[str, Any],
    tone_controls: Dict[str, float],
    wb: Dict[str, Any],
    color: Dict[str, Any],
    detail: Dict[str, Any],
) -> Dict[str, Any]:

    exposure_value = _safe_float_value(
        exposure,
        "value",
        0.0,
    )

    exposure_confidence = _safe_normalized_confidence(
        exposure.get(
            "confidence",
            0.0,
        )
    )

    wb_confidence = _safe_normalized_confidence(
        wb.get(
            "confidence",
            0.0,
        )
    )

    wb_reliability = _get_wb_reliability(
        wb
    )

    # ------------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------------

    exposure_strength = _exposure_strength_from_confidence(
        exposure_confidence
    )

    exposure_value *= exposure_strength

    # Prevent insignificant micro-adjustments.
    if abs(exposure_value) < 0.04:
        exposure_value = 0.0

    # ------------------------------------------------------------------------
    # White balance
    # ------------------------------------------------------------------------

    gains = _get_safe_wb_gains(
        wb
    )

    wb_strength = _wb_strength_from_confidence(
        wb_reliability
    )

    if wb_strength <= 0.0:
        gains = np.ones(
            3,
            dtype=np.float32,
        )

    else:
        log_gains = np.log(
            np.maximum(
                gains,
                EPS,
            )
        )

        gains = normalize_rgb_gains(
            np.exp(
                log_gains
                * wb_strength
            )
        )

    # The WB module owns the conversion from gains
    # to its existing compatibility representation.
    temperature_cast, tint_cast = (
        gains_to_temperature_tint(
            gains
        )
    )

    if wb_strength < 0.25:
        temperature_cast = 0.0
        tint_cast = 0.0

    if abs(
        temperature_cast
    ) < 0.35:
        temperature_cast = 0.0

    if abs(
        tint_cast
    ) < 0.35:
        tint_cast = 0.0

    # ------------------------------------------------------------------------
    # Sharpening
    # ------------------------------------------------------------------------

    quality = str(
        detail.get(
            "quality",
            "normal",
        )
    ).lower()

    if quality == "very_soft":
        sharpen = 0.20

    elif quality == "soft":
        sharpen = 0.12

    elif quality == "normal":
        sharpen = 0.06

    elif quality == "sharp":
        sharpen = 0.02

    else:
        sharpen = 0.0

    # If noise is explicitly reported, reduce sharpening.
    noise_ratio = _safe_float_value(
        detail,
        "noise_ratio",
        0.0,
    )

    if noise_ratio > 0.18:
        sharpen *= 0.65

    if noise_ratio > 0.28:
        sharpen *= 0.40

    # ------------------------------------------------------------------------
    # Color controls
    # ------------------------------------------------------------------------

    vibrance = _safe_float_value(
        color,
        "vibrance",
        0.0,
    )

    saturation = _safe_float_value(
        color,
        "saturation",
        0.0,
    )

    return {
        "exposure": round(
            clamp(
                exposure_value,
                -2.0,
                2.0,
            ),
            4,
        ),

        "contrast": safe_float(
            tone_controls.get(
                "contrast",
                0.0,
            ),
            0.0,
        ),

        "highlights": safe_float(
            tone_controls.get(
                "highlights",
                0.0,
            ),
            0.0,
        ),

        "shadows": safe_float(
            tone_controls.get(
                "shadows",
                0.0,
            ),
            0.0,
        ),

        "whites": safe_float(
            tone_controls.get(
                "whites",
                0.0,
            ),
            0.0,
        ),

        "blacks": safe_float(
            tone_controls.get(
                "blacks",
                0.0,
            ),
            0.0,
        ),

        "temperature": round(
            safe_float(
                temperature_cast,
                0.0,
            ),
            4,
        ),

        "tint": round(
            safe_float(
                tint_cast,
                0.0,
            ),
            4,
        ),

        "temperature_gains_rgb": [
            round(
                safe_float(
                    x,
                    1.0,
                ),
                6,
            )
            for x in gains
        ],

        "vibrance": round(
            vibrance,
            4,
        ),

        "saturation": round(
            saturation,
            4,
        ),

        "noise_reduction": 0.0,

        "sharpen": round(
            clamp(
                sharpen,
                0.0,
                0.25,
            ),
            4,
        ),
    }


# ============================================================================
# FINAL CORRECTION PLAN
# ============================================================================

def derive_correction_plan(
    exposure: Dict[str, Any],
    tone_controls: Dict[str, float],
    wb: Dict[str, Any],
    color: Dict[str, Any],
) -> Dict[str, Any]:

    # ------------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------------

    exposure_confidence = _safe_normalized_confidence(
        exposure.get(
            "confidence",
            0.0,
        )
    )

    exposure_value = _safe_float_value(
        exposure,
        "value",
        0.0,
    )

    exposure_strength = _exposure_strength_from_confidence(
        exposure_confidence
    )

    exposure_value *= exposure_strength

    if abs(
        exposure_value
    ) < 0.04:
        exposure_value = 0.0

    exposure_value = clamp(
        exposure_value,
        -2.0,
        2.0,
    )

    # ------------------------------------------------------------------------
    # White balance
    # ------------------------------------------------------------------------

    wb_confidence = _safe_normalized_confidence(
        wb.get(
            "confidence",
            0.0,
        )
    )

    wb_reliability = _get_wb_reliability(
        wb
    )

    raw_gains = _get_safe_wb_gains(
        wb
    )

    wb_strength = _wb_strength_from_confidence(
        wb_reliability
    )

    if wb_strength <= 0.0:
        gains = np.ones(
            3,
            dtype=np.float32,
        )

    else:
        raw_logs = np.log(
            np.maximum(
                raw_gains,
                EPS,
            )
        )

        gains = normalize_rgb_gains(
            np.exp(
                raw_logs
                * wb_strength
            )
        )

    final_temperature, final_tint = (
        gains_to_temperature_tint(
            gains
        )
    )

    if wb_strength < 0.25:
        final_temperature = 0.0
        final_tint = 0.0

    if abs(
        final_temperature
    ) < 0.35:
        final_temperature = 0.0

    if abs(
        final_tint
    ) < 0.35:
        final_tint = 0.0

    # ------------------------------------------------------------------------
    # Tone
    # ------------------------------------------------------------------------

    contrast = safe_float(
        tone_controls.get(
            "contrast",
            0.0,
        ),
        0.0,
    )

    highlights = safe_float(
        tone_controls.get(
            "highlights",
            0.0,
        ),
        0.0,
    )

    shadows = safe_float(
        tone_controls.get(
            "shadows",
            0.0,
        ),
        0.0,
    )

    whites = safe_float(
        tone_controls.get(
            "whites",
            0.0,
        ),
        0.0,
    )

    blacks = safe_float(
        tone_controls.get(
            "blacks",
            0.0,
        ),
        0.0,
    )

    # Clamp compatibility output so a faulty upstream calculation
    # cannot generate extreme edits.
    contrast = clamp(
        contrast,
        -100.0,
        100.0,
    )

    highlights = clamp(
        highlights,
        -100.0,
        100.0,
    )

    shadows = clamp(
        shadows,
        -100.0,
        100.0,
    )

    whites = clamp(
        whites,
        -100.0,
        100.0,
    )

    blacks = clamp(
        blacks,
        -100.0,
        100.0,
    )

    # ------------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------------

    vibrance = clamp(
        _safe_float_value(
            color,
            "vibrance",
            0.0,
        ),
        -100.0,
        100.0,
    )

    saturation = clamp(
        _safe_float_value(
            color,
            "saturation",
            0.0,
        ),
        -100.0,
        100.0,
    )

    return {
        "exposure_ev": round(
            exposure_value,
            4,
        ),

        "exposure_confidence": round(
            exposure_confidence,
            4,
        ),

        "white_balance": {
            "gains_rgb": [
                round(
                    safe_float(
                        x,
                        1.0,
                    ),
                    6,
                )
                for x in gains
            ],

            "confidence": round(
                wb_confidence,
                4,
            ),

            "reliability": round(
                wb_reliability,
                4,
            ),

            "strength": round(
                wb_strength,
                4,
            ),

            "temperature_cast": round(
                safe_float(
                    final_temperature,
                    0.0,
                ),
                4,
            ),

            "tint_cast": round(
                safe_float(
                    final_tint,
                    0.0,
                ),
                4,
            ),
        },

        "light": {
            "contrast": round(
                contrast,
                4,
            ),

            "highlights": round(
                highlights,
                4,
            ),

            "shadows": round(
                shadows,
                4,
            ),

            "whites": round(
                whites,
                4,
            ),

            "blacks": round(
                blacks,
                4,
            ),
        },

        "color": {
            "vibrance": round(
                vibrance,
                4,
            ),

            "saturation": round(
                saturation,
                4,
            ),
        },
    }


# ============================================================================
# IMAGE QUALITY
# ============================================================================

def analyze_quality(
    image: np.ndarray,
    tone: Dict[str, Any],
    detail: Dict[str, Any],
) -> Dict[str, Any]:

    if image is None:
        raise ValueError(
            "image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be numpy.ndarray."
        )

    if image.ndim < 2:
        raise ValueError(
            "image must have at least two dimensions."
        )

    h, w = image.shape[:2]

    clipping = tone.get(
        "clipping",
        {},
    )

    highlight_clip = clamp(
        _safe_float_value(
            clipping,
            "highlight_fraction",
            0.0,
        ),
        0.0,
        1.0,
    )

    shadow_clip = clamp(
        _safe_float_value(
            clipping,
            "shadow_fraction",
            0.0,
        ),
        0.0,
        1.0,
    )

    sharpness = max(
        0.0,
        _safe_float_value(
            detail,
            "sharpness",
            0.0,
        ),
    )

    noise_ratio = clamp(
        _safe_float_value(
            detail,
            "noise_ratio",
            0.0,
        ),
        0.0,
        1.0,
    )

    edge_fraction = clamp(
        _safe_float_value(
            detail,
            "edge_fraction",
            0.0,
        ),
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Individual quality components
    # ------------------------------------------------------------------------

    clipping_penalty = (
        clamp(
            highlight_clip * 12.0,
            0.0,
            0.35,
        )
        +
        clamp(
            shadow_clip * 7.0,
            0.0,
            0.25,
        )
    )

    # Sharpness is image-dependent, so use the detail module's
    # categorical quality when available rather than treating one
    # absolute Laplacian value as universal.
    detail_quality = str(
        detail.get(
            "quality",
            "normal",
        )
    ).lower()

    detail_penalty = 0.0

    if detail_quality == "very_soft":
        detail_penalty = 0.22

    elif detail_quality == "soft":
        detail_penalty = 0.10

    elif detail_quality == "normal":
        detail_penalty = 0.02

    elif detail_quality == "sharp":
        detail_penalty = 0.0

    # Noise is a quality problem, but don't punish high-ISO images
    # excessively because legitimate texture can resemble noise.
    noise_penalty = clamp(
        noise_ratio * 0.45,
        0.0,
        0.20,
    )

    # Very little edge information can indicate blur,
    # but edge_fraction alone is not enough to call an image bad.
    edge_penalty = 0.0

    if edge_fraction < 0.015:
        edge_penalty = 0.08

    elif edge_fraction < 0.008:
        edge_penalty = 0.14

    # ------------------------------------------------------------------------
    # Final quality score
    # ------------------------------------------------------------------------

    quality = (
        1.0
        - clipping_penalty
        - detail_penalty
        - noise_penalty
        - edge_penalty
    )

    quality = clamp(
        quality,
        0.0,
        1.0,
    )

    # Human-readable state.
    if quality >= 0.85:
        state = "excellent"

    elif quality >= 0.70:
        state = "good"

    elif quality >= 0.50:
        state = "fair"

    elif quality >= 0.30:
        state = "poor"

    else:
        state = "very_poor"

    return {
        "width": int(w),

        "height": int(h),

        "megapixels": round(
            w * h
            / 1_000_000.0,
            3,
        ),

        "score": round(
            quality,
            4,
        ),

        "state": state,

        "components": {
            "clipping_penalty": round(
                clipping_penalty,
                4,
            ),

            "detail_penalty": round(
                detail_penalty,
                4,
            ),

            "noise_penalty": round(
                noise_penalty,
                4,
            ),

            "edge_penalty": round(
                edge_penalty,
                4,
            ),
        },
    }