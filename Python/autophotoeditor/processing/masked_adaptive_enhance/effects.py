from __future__ import annotations

from .core import *
from .analysis import *
from .color import *

def apply_tonal_adjustment(
    roi: np.ndarray,
    exposure: float,
    shadow_lift: float,
    highlight_reduction: float,
    contrast_amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0] * (
        255.0 / 255.0
    )

    # ------------------------------------------------------------------------
    # Exposure.
    # ------------------------------------------------------------------------

    if abs(exposure) > 0.001:

        L += (
            exposure
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Shadow lift.
    # ------------------------------------------------------------------------

    if shadow_lift > 0:

        normalized = L / 255.0

        shadow_weight = np.clip(
            (0.42 - normalized)
            / 0.42,
            0.0,
            1.0,
        )

        shadow_weight = (
            shadow_weight
            ** 1.35
        )

        L += (
            shadow_weight
            * shadow_lift
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Highlight compression.
    # ------------------------------------------------------------------------

    if highlight_reduction > 0:

        normalized = L / 255.0

        highlight_weight = np.clip(
            (normalized - 0.68)
            / 0.32,
            0.0,
            1.0,
        )

        highlight_weight = (
            highlight_weight
            ** 1.20
        )

        L -= (
            highlight_weight
            * highlight_reduction
            * profile.tone
        )

    # ------------------------------------------------------------------------
    # Adaptive contrast.
    # ------------------------------------------------------------------------

    if contrast_amount > 0:

        normalized = L / 255.0

        # Very gentle S curve.
        curve = (
            normalized
            + contrast_amount
            * (
                normalized
                - normalized * normalized
            )
        )

        L = (
            normalized * 255.0
            * 0.78
            +
            curve * 255.0
            * 0.22
        )

    L = np.clip(
        L,
        0,
        255,
    )

    lab[:, :, 0] = L

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# LOCAL CONTRAST
# ============================================================================

def apply_local_clarity(
    roi: np.ndarray,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.clarity_strength
        * profile.clarity
    )

    if amount <= 0:
        return roi.copy()

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0]

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        CONFIG.clarity_sigma,
    )

    detail = L - blur

    # Prevent strong clarity on extreme shadows/highlights.
    normalized = L / 255.0

    protection = (
        1.0
        -
        0.65
        *
        np.maximum(
            np.clip(
                0.12 - normalized,
                0,
                0.12,
            ) / 0.12,
            np.clip(
                normalized - 0.92,
                0,
                0.08,
            ) / 0.08,
        )
    )

    L += (
        detail
        * amount
        * protection
    )

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


# ============================================================================
# VIBRANCE
# ============================================================================

def apply_vibrance(
    roi: np.ndarray,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.saturation_strength
        * profile.saturation
    )

    if amount <= 0:
        return roi.copy()

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float32)

    saturation = (
        hsv[:, :, 1]
        / 255.0
    )

    # Less saturated pixels get more boost.
    protection = (
        1.0
        - saturation
    )

    # Protect already highly saturated colors.
    boost = (
        1.0
        + amount
        * protection
    )

    if profile.skin:

        # Skin receives only a tiny saturation adjustment.
        boost = (
            1.0
            + (
                boost - 1.0
            ) * 0.35
        )

    hsv[:, :, 1] *= boost

    # Hard saturation protection.
    if profile.saturation_limit < 1.0:

        max_sat = (
            255.0
            * (
                0.75
                + 0.25
                * profile.saturation_limit
            )
        )

        hsv[:, :, 1] = np.minimum(
            hsv[:, :, 1],
            max_sat,
        )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255,
    )

    return cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )


# ============================================================================
# LUMINANCE SHARPENING
# ============================================================================

def apply_luminance_sharpening(
    roi: np.ndarray,
    sharpness: float,
    amount: float,
    profile: MaskProfile,
) -> np.ndarray:

    if amount <= 0:
        return roi.copy()

    amount *= (
        CONFIG.sharpening_strength
        * profile.sharpening
    )

    amount = float(
        np.clip(
            amount,
            0.0,
            CONFIG.max_sharpen,
        )
    )

    if amount <= 0:
        return roi.copy()

    # Already sharp images receive less sharpening.
    if sharpness > 4500:
        amount *= 0.20

    elif sharpness > 2500:
        amount *= 0.40

    elif sharpness > 1200:
        amount *= 0.70

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0]

    if sharpness < 600:

        sigma = 0.85

    elif sharpness < 1400:

        sigma = 1.00

    else:

        sigma = 1.15

    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        sigma,
    )

    detail = L - blur

    # Don't sharpen extreme highlights.
    normalized = L / 255.0

    protection = 1.0 - 0.55 * np.clip(
        (normalized - 0.88)
        / 0.12,
        0.0,
        1.0,
    )

    if profile.skin:
        amount *= 0.55

    L += (
        detail
        * amount
        * protection
    )

    lab[:, :, 0] = np.clip(
        L,
        0,
        255,
    )

    return cv2.cvtColor(
        lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


