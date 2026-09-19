from __future__ import annotations

from .core import *
from .color import *

def apply_exposure(
    image: np.ndarray,
    exposure: float,
) -> np.ndarray:

    if abs(exposure) < EPS:
        return image

    # Exposure is measured in stops.
    multiplier = 2.0 ** exposure

    linear = srgb_to_linear(image)

    linear *= multiplier

    return clamp01(
        linear_to_srgb(linear)
    )


# ============================================================================
# WHITE BALANCE
# ============================================================================

def apply_white_balance(
    image: np.ndarray,
    temperature: float,
    tint: float,
) -> np.ndarray:

    if abs(temperature) < EPS and abs(tint) < EPS:
        return image

    result = image.copy()

    # Conservative approximation.
    #
    # Lightroom's temperature scale is not a simple RGB multiplier.
    # These small coefficients prevent extreme color casts.

    temp_strength = temperature / 100.0
    tint_strength = tint / 100.0

    result[..., 0] *= (
        1.0 + temp_strength * 0.10
    )

    result[..., 2] *= (
        1.0 - temp_strength * 0.10
    )

    result[..., 1] *= (
        1.0 + tint_strength * 0.04
    )

    result[..., 0] *= (
        1.0 - tint_strength * 0.02
    )

    result[..., 2] *= (
        1.0 - tint_strength * 0.02
    )

    return clamp01(result)


# ============================================================================
# LUMINANCE
# ============================================================================

def perceptual_luminance(
    image: np.ndarray,
) -> np.ndarray:

    # Rec.709-like luminance.
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)


# ============================================================================
# TONE
# ============================================================================

def apply_tone(
    image: np.ndarray,
    tone: ToneSettings,
) -> np.ndarray:

    result = image.copy()

    # ------------------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------------------

    if abs(tone.contrast) > EPS:

        strength = tone.contrast / 100.0

        # Smooth contrast approximation.
        factor = 1.0 + strength * 1.5

        result = (
            (result - 0.5) * factor
            + 0.5
        )

    # ------------------------------------------------------------------------
    # Highlights / Shadows
    # ------------------------------------------------------------------------

    luminance = perceptual_luminance(
        result
    )

    if abs(tone.highlights) > EPS:

        amount = tone.highlights / 100.0

        # Highlight mask.
        mask = smoothstep(
            0.45,
            0.95,
            luminance,
        )

        # Preserve shadows.
        influence = mask * mask

        result += (
            amount
            * 0.22
            * influence[..., None]
        )

    if abs(tone.shadows) > EPS:

        amount = tone.shadows / 100.0

        # Shadow mask.
        mask = 1.0 - smoothstep(
            0.05,
            0.60,
            luminance,
        )

        influence = mask * mask

        result += (
            amount
            * 0.22
            * influence[..., None]
        )

    # ------------------------------------------------------------------------
    # Whites
    # ------------------------------------------------------------------------

    if abs(tone.whites) > EPS:

        amount = tone.whites / 100.0

        mask = smoothstep(
            0.72,
            1.00,
            luminance,
        )

        result += (
            amount
            * 0.18
            * mask[..., None]
        )

    # ------------------------------------------------------------------------
    # Blacks
    # ------------------------------------------------------------------------

    if abs(tone.blacks) > EPS:

        amount = tone.blacks / 100.0

        mask = 1.0 - smoothstep(
            0.00,
            0.30,
            luminance,
        )

        result += (
            amount
            * 0.18
            * mask[..., None]
        )

    return clamp01(result)


# ============================================================================
# POINT CURVE
# ============================================================================

def apply_point_curve(
    image: np.ndarray,
    curve: PointCurve,
) -> np.ndarray:

    if len(curve.points) < 2:
        return image

    points = curve.points

    xs = np.array(
        [p[0] for p in points],
        dtype=np.float32,
    )

    ys = np.array(
        [p[1] for p in points],
        dtype=np.float32,
    )

    # Guarantee endpoints.
    if xs[0] > 0.0:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, ys[0])

    if xs[-1] < 1.0:
        xs = np.append(xs, 1.0)
        ys = np.append(ys, ys[-1])

    result = np.interp(
        image,
        xs,
        ys,
    )

    return clamp01(
        result.astype(np.float32)
    )


# ============================================================================
# HSL
# ============================================================================

def parse_hsl_override(
    value: str,
) -> HSLAdjustment:

    parts = [
        x.strip()
        for x in value.split(",")
    ]

    if len(parts) != 3:
        raise ValueError(
            "HSL override must be "
            "hue,saturation,luminance"
        )

    return HSLAdjustment(
        hue=safe_float(parts[0]),
        saturation=safe_float(parts[1]),
        luminance=safe_float(parts[2]),
    )


def hsl_channel_mask(
    hue: np.ndarray,
    center: float,
) -> np.ndarray:

    distance = angular_distance(
        hue,
        center,
    )

    # Lightroom's channel ranges overlap.
    # We use a smooth bell-like response rather
    # than a hard triangular edge.

    width = 38.0

    mask = np.clip(
        1.0 - distance / width,
        0.0,
        1.0,
    )

    # Smooth the transition.
    mask = mask * mask * (
        3.0 - 2.0 * mask
    )

    return mask.astype(np.float32)


def apply_hsl(
    image: np.ndarray,
    hsl: Dict[str, HSLAdjustment],
) -> np.ndarray:

    if not hsl:
        return image

    result = image.copy()

    L, C, H = rgb_to_oklch(
        result
    )

    original_H = H.copy()
    original_C = C.copy()

    # Saturation threshold.
    #
    # HSL adjustments should have little effect on
    # near-gray pixels.
    saturation_weight = np.clip(
        original_C / 0.18,
        0.0,
        1.0,
    )

    total_hue_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_sat_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_lum_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_weight = np.zeros_like(
        H,
        dtype=np.float32,
    )

    # IMPORTANT:
    # Every mask is calculated from ORIGINAL hue.
    #
    # This prevents:
    #
    # Green -> Yellow
    # then the modified Yellow -> Orange
    #
    # cross-talk that causes residue/artifacts.
    for channel in HSL_CHANNELS:

        adjustment = hsl.get(
            channel,
            HSLAdjustment(),
        )

        if (
            abs(adjustment.hue) < EPS
            and abs(adjustment.saturation) < EPS
            and abs(adjustment.luminance) < EPS
        ):
            continue

        center = HSL_CENTERS[channel]

        mask = hsl_channel_mask(
            original_H,
            center,
        )

        mask *= saturation_weight

        total_weight += mask

        # --------------------------------------------------------------------
        # Hue
        # --------------------------------------------------------------------

        if abs(adjustment.hue) > EPS:

            # Lightroom HSL hue is approximately -100..+100.
            #
            # It is NOT a literal degree value.
            #
            # Use a conservative maximum shift.
            hue_shift = (
                adjustment.hue
                * 0.45
            )

            total_hue_shift += (
                mask * hue_shift
            )

        # --------------------------------------------------------------------
        # Saturation
        # --------------------------------------------------------------------

        if abs(adjustment.saturation) > EPS:

            # Scale chroma rather than raw RGB saturation.
            saturation_change = (
                adjustment.saturation
                / 100.0
            )

            total_sat_shift += (
                mask * saturation_change
            )

        # --------------------------------------------------------------------
        # Luminance
        # --------------------------------------------------------------------

        if abs(adjustment.luminance) > EPS:

            luminance_change = (
                adjustment.luminance
                / 100.0
            )

            total_lum_shift += (
                mask * luminance_change
            )

    # Normalize overlapping channels.
    normalization = np.maximum(
        total_weight,
        1.0,
    )

    hue_shift = (
        total_hue_shift
        / normalization
    )

    sat_shift = (
        total_sat_shift
        / normalization
    )

    lum_shift = (
        total_lum_shift
        / normalization
    )

    # ------------------------------------------------------------------------
    # Apply hue
    # ------------------------------------------------------------------------

    H = normalize_angle_degrees(
        original_H + hue_shift
    )

    # ------------------------------------------------------------------------
    # Apply saturation
    # ------------------------------------------------------------------------

    # Chroma scale.
    C = C * (
        1.0 + sat_shift * 0.65
    )

    C = np.maximum(
        C,
        0.0,
    )

    # ------------------------------------------------------------------------
    # Apply luminance
    # ------------------------------------------------------------------------

    L = L + (
        lum_shift * 0.16
    )

    L = np.clip(
        L,
        0.0,
        1.0,
    )

    result = oklch_to_rgb(
        L,
        C,
        H,
    )

    return clamp01(result)


# ============================================================================
# GLOBAL COLOR
# ============================================================================

def apply_color(
    image: np.ndarray,
    color: ColorSettings,
) -> np.ndarray:

    result = image.copy()

    # ------------------------------------------------------------------------
    # Global saturation
    # ------------------------------------------------------------------------

    if abs(color.saturation) > EPS:

        L, C, H = rgb_to_oklch(
            result
        )

        amount = color.saturation / 100.0

        C *= (
            1.0 + amount * 0.70
        )

        C = np.maximum(
            C,
            0.0,
        )

        result = oklch_to_rgb(
            L,
            C,
            H,
        )

    # ------------------------------------------------------------------------
    # Vibrance
    # ------------------------------------------------------------------------

    if abs(color.vibrance) > EPS:

        L, C, H = rgb_to_oklch(
            result
        )

        amount = color.vibrance / 100.0

        # Stronger on moderately saturated pixels.
        #
        # Very saturated colors receive less effect.
        saturation_factor = np.clip(
            1.0 - C / 0.30,
            0.0,
            1.0,
        )

        # Do not boost almost-gray pixels aggressively.
        lower_bound = smoothstep(
            0.015,
            0.05,
            C,
        )

        weight = (
            saturation_factor
            * lower_bound
        )

        C *= (
            1.0
            + amount
            * 0.75
            * weight
        )

        C = np.maximum(
            C,
            0.0,
        )

        result = oklch_to_rgb(
            L,
            C,
            H,
        )

    return clamp01(result)


# ============================================================================
# CLARITY
# ============================================================================

def apply_clarity(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    # Work in luminance to reduce color halos.
    L = perceptual_luminance(
        image
    )

    L8 = np.round(
        np.clip(L, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    sigma = 2.0

    blurred = cv2.GaussianBlur(
        L8,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    ).astype(np.float32) / 255.0

    detail = L - blurred

    enhanced = (
        L
        + detail
        * strength
        * 0.9
    )

    delta = enhanced - L

    result = image + delta[..., None]

    return clamp01(result)


# ============================================================================
# TEXTURE
# ============================================================================

def apply_texture(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    L = perceptual_luminance(
        image
    )

    # Small-scale detail.
    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=0.8,
        sigmaY=0.8,
    )

    detail = L - blur

    result = (
        image
        + detail[..., None]
        * strength
        * 0.45
    )

    return clamp01(result)


# ============================================================================
# DEHAZE
# ============================================================================

def apply_dehaze(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    result = image.copy()

    # Conservative approximation:
    #
    # Positive dehaze:
    #   increase local contrast
    #   slightly deepen blacks
    #
    # Negative dehaze:
    #   reduce local contrast
    #   lift blacks

    L = perceptual_luminance(
        result
    )

    blurred = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=8.0,
        sigmaY=8.0,
    )

    local = L - blurred

    result += (
        local[..., None]
        * strength
        * 0.55
    )

    result *= (
        1.0
        - strength * 0.06
    )

    result += (
        strength * 0.025
    )

    return clamp01(result)


def apply_effects(
    image: np.ndarray,
    effects: EffectsSettings,
) -> np.ndarray:

    result = image

    if abs(effects.texture) > EPS:
        result = apply_texture(
            result,
            effects.texture,
        )

    if abs(effects.clarity) > EPS:
        result = apply_clarity(
            result,
            effects.clarity,
        )

    if abs(effects.dehaze) > EPS:
        result = apply_dehaze(
            result,
            effects.dehaze,
        )

    return clamp01(result)


# ============================================================================
# NOISE REDUCTION
# ============================================================================

def apply_noise_reduction(
    image: np.ndarray,
    settings: NoiseSettings,
) -> np.ndarray:

    if (
        abs(settings.luminance) < EPS
        and abs(settings.color) < EPS
    ):
        return image

    result = image.copy()

    strength = max(
        settings.luminance,
        settings.color,
    )

    if strength <= 0:
        return result

    # OpenCV bilateral filter gives a reasonably safe
    # photographic noise reduction without destroying
    # edges as aggressively as Gaussian blur.

    diameter = 5

    sigma_color = (
        10.0
        + strength * 0.7
    )

    sigma_space = 2.0

    bgr = cv2.cvtColor(
        np.round(
            result * 255.0
        ).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    filtered = cv2.bilateralFilter(
        bgr,
        diameter,
        sigma_color,
        sigma_space,
    )

    filtered = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    blend = np.clip(
        strength / 100.0,
        0.0,
        1.0,
    )

    result = (
        result * (1.0 - blend)
        + filtered * blend
    )

    return clamp01(result)


# ============================================================================
# SHARPENING
# ============================================================================

def apply_sharpening(
    image: np.ndarray,
    settings: SharpenSettings,
) -> np.ndarray:

    if settings.amount <= 0:
        return image

    amount = np.clip(
        settings.amount / 100.0,
        0.0,
        3.0,
    )

    radius = np.clip(
        settings.radius,
        0.2,
        5.0,
    )

    L = perceptual_luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
    )

    detail = L - blurred

    # Masking:
    # Lightroom masking suppresses sharpening in
    # relatively flat areas.

    gradient_x = cv2.Sobel(
        L,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        L,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = np.sqrt(
        gradient_x * gradient_x
        + gradient_y * gradient_y
    )

    gradient /= (
        np.max(gradient)
        + EPS
    )

    masking = np.clip(
        settings.masking / 100.0,
        0.0,
        1.0,
    )

    mask = (
        gradient
        * (1.0 - masking)
        + masking
    )

    sharpened_luminance = (
        L
        + detail
        * amount
        * mask
    )

    delta = (
        sharpened_luminance
        - L
    )

    result = (
        image
        + delta[..., None]
    )

    return clamp01(result)


