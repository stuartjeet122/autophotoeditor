from __future__ import annotations

from .core import *
from .analysis import *

def apply_neutral_color(
    roi: np.ndarray,
    gains: Tuple[float, float, float],
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    # Skin gets much weaker color correction.
    if profile.skin:
        amount *= 0.35

    if profile.reflective:
        amount *= 0.18

    amount *= profile.wb

    if amount <= 0:
        return roi.copy()

    b_gain = np.clip(
        1.0 + (gains[2] - 1.0) * amount,
        0.90,
        1.10,
    )

    g_gain = np.clip(
        1.0 + (gains[1] - 1.0) * amount,
        0.90,
        1.10,
    )

    r_gain = np.clip(
        1.0 + (gains[0] - 1.0) * amount,
        0.90,
        1.10,
    )

    luma_gain = (
        0.114 * b_gain
        + 0.587 * g_gain
        + 0.299 * r_gain
    )

    if luma_gain > 1e-6:
        b_gain /= luma_gain
        g_gain /= luma_gain
        r_gain /= luma_gain

    src = roi.astype(
        np.float32
    )

    result = src.copy()

    result[:, :, 0] *= b_gain
    result[:, :, 1] *= g_gain
    result[:, :, 2] *= r_gain

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# LAB NEUTRAL CAST PROTECTION
# ============================================================================

def apply_subtle_lab_neutralization(
    roi: np.ndarray,
    local_stats: Dict[str, float],
    ring_stats: Dict[str, float],
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    # This is NOT background color copying.
    #
    # It only removes a very small detected chromatic bias.
    da = (
        128.0
        - ring_stats.get(
            "lab_a",
            128.0,
        )
    )

    db = (
        128.0
        - ring_stats.get(
            "lab_b",
            128.0,
        )
    )

    # If surrounding area itself is almost neutral,
    # there is little evidence for a color cast.
    magnitude = math.sqrt(
        da * da
        + db * db
    )

    if magnitude < 1.5:
        return roi.copy()

    strength = (
        CONFIG.color_match_strength
        * profile.wb
        * amount
    )

    if profile.skin:
        strength *= 0.25

    if profile.reflective:
        strength *= 0.15

    da *= strength
    db *= strength

    da = float(
        np.clip(
            da,
            -CONFIG.max_a_change,
            CONFIG.max_a_change,
        )
    )

    db = float(
        np.clip(
            db,
            -CONFIG.max_b_change,
            CONFIG.max_b_change,
        )
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    lab[:, :, 1] += da
    lab[:, :, 2] += db

    lab[:, :, 1] = np.clip(
        lab[:, :, 1],
        0,
        255,
    )

    lab[:, :, 2] = np.clip(
        lab[:, :, 2],
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# EXPOSURE
# ============================================================================

def calculate_exposure(
    target: Dict[str, float],
    surrounding: Dict[str, float],
    profile: MaskProfile,
) -> float:

    target_p50 = target.get(
        "p50",
        target.get(
            "brightness",
            0.5,
        ),
    )

    surrounding_p50 = surrounding.get(
        "p50",
        surrounding.get(
            "brightness",
            target_p50,
        ),
    )

    difference = (
        surrounding_p50
        - target_p50
    ) * 255.0

    if abs(difference) <= CONFIG.exposure_dead_zone:
        return 0.0

    correction = (
        difference
        * CONFIG.exposure_match_strength
        * profile.exposure
    )

    if correction < 0:

        if profile.skin:
            correction *= (
                CONFIG.skin_negative_exposure_factor
            )
        else:
            correction *= 0.55

    else:

        if profile.skin:
            correction *= (
                CONFIG.skin_positive_exposure_factor
            )

    return float(
        np.clip(
            correction,
            -CONFIG.max_exposure_change * 0.45,
            CONFIG.max_exposure_change,
        )
    )


# ============================================================================
# SHADOW RECOVERY
# ============================================================================

def calculate_shadow_lift(
    stats: Dict[str, float],
    profile: MaskProfile,
) -> float:

    shadow_fraction = stats.get(
        "shadow_fraction",
        0.0,
    )

    p25 = stats.get(
        "p25",
        0.25,
    )

    danger = 0.0

    if p25 < 0.28:
        danger += (
            0.28 - p25
        ) / 0.28

    danger += (
        shadow_fraction
        * 1.25
    )

    danger = float(
        np.clip(
            danger,
            0.0,
            1.5,
        )
    )

    amount = (
        danger
        * CONFIG.shadow_strength
        * profile.recovery
        * CONFIG.max_shadow_lift
    )

    if profile.skin:
        amount *= 0.65

    return float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_shadow_lift,
        )
    )


# ============================================================================
# HIGHLIGHT RECOVERY
# ============================================================================

def calculate_highlight_reduction(
    stats: Dict[str, float],
    profile: MaskProfile,
) -> float:

    p90 = stats.get(
        "p90",
        0.8,
    )

    clipped = stats.get(
        "clipped_fraction",
        0.0,
    )

    danger = 0.0

    if p90 > 0.82:
        danger += (
            p90 - 0.82
        ) / 0.18

    danger += clipped * 3.0

    danger = float(
        np.clip(
            danger,
            0.0,
            1.5,
        )
    )

    amount = (
        danger
        * CONFIG.highlight_strength
        * profile.recovery
        * CONFIG.max_highlight_reduction
    )

    # Skin highlights are protected but not ignored.
    if profile.skin:
        amount *= 0.65

    return float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_highlight_reduction,
        )
    )


