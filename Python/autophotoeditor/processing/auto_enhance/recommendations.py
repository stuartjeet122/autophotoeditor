from __future__ import annotations

from .core import *
from .analysis import *

def derive_exposure_recommendation(
    tone: Mapping[str, Any],
    scene: Mapping[str, Any],
    subject_confidence: float,
) -> Estimate:
    """
    Evidence-driven exposure recommendation.

    Important design rule: the analyzer must not force every photograph toward
    one global luminance target. Exposure is inferred from the distribution of
    midtones, subject brightness, highlight headroom and clipping. Scene labels
    are used only as weak context.
    """
    linear = tone["linear_light"]
    clip = tone["clipping"]
    perceptual = tone["perceptual"]

    global_ev = float(linear["global_median_ev_from_18pct"])
    subject_ev = float(linear["subject_median_ev_from_18pct"])
    salient_ev = float(linear["saliency_weighted_ev_from_18pct"])
    subject_cov = float(linear["subject_coverage_percentage"]) / 100.0

    p10 = float(perceptual["percentiles_lab"].get("p10", 0.0))
    p25 = float(perceptual["percentiles_lab"].get("p25", 0.0))
    p50 = float(perceptual["percentiles_lab"].get("p50", 0.0))
    p75 = float(perceptual["percentiles_lab"].get("p75", 0.0))
    p90 = float(perceptual["percentiles_lab"].get("p90", 0.0))
    p95 = float(perceptual["percentiles_lab"].get("p95", 0.0))

    high_clip = float(clip["highlight_all_channels_percentage"])
    any_high_clip = float(clip["highlight_any_channel_percentage"])
    low_clip = float(clip["shadow_all_channels_percentage"])
    dark_pct = float(perceptual["dark_pixels_percentage"])
    bright_pct = float(perceptual["bright_pixels_percentage"])
    headroom = float(linear["highlight_headroom_ev_at_p99"])

    # Midtone evidence.  The middle 25-75% distribution is more stable than
    # the image mean and is much less affected by sky, lamps and black borders.
    mid_l = 0.55 * p50 + 0.225 * p25 + 0.225 * p75

    # Start from a neutral photographic midpoint, but make the target adaptive.
    # This is intentionally a range rather than a single fixed scene target.
    # For normal images the desired LAB L median is roughly 48-58.
    target_mid = 53.0

    # Preserve obvious high-key / low-key intent only when the distribution
    # strongly supports it.  A scene classification alone is never sufficient.
    scene_scores = scene.get("scores", {}) if isinstance(scene, Mapping) else {}
    high_key_score = float(scene_scores.get("high_key", 0.0))
    low_key_score = float(scene_scores.get("low_key", 0.0))
    if high_key_score >= 0.72 and p25 >= 45 and p50 >= 58:
        target_mid = 61.0
    elif low_key_score >= 0.72 and p75 <= 58 and p50 <= 43:
        target_mid = 42.0

    # Convert the perceptual midpoint error to EV. Around middle gray, a
    # 6-7 L-point shift is approximately a quarter stop. Keep it deliberately
    # gentle because the enhancer will apply the final curve.
    mid_error = target_mid - mid_l
    perceptual_ev = clamp(mid_error / 25.0, -1.0, 1.0)

    # Subject evidence is trusted when a meaningful subject mask exists.
    # Portraits should not be exposed from the background.
    subject_weight = clamp(subject_confidence, 0.0, 1.0) * clamp(subject_cov / 0.08, 0.0, 1.0)
    subject_ev_target = -0.15
    subject_delta = clamp(subject_ev_target - subject_ev, -1.0, 1.0)

    salient_delta = clamp(-0.10 - salient_ev, -0.8, 0.8)

    # Blend independent evidence. Subject dominates only when it is reliable.
    if subject_weight > 0.20:
        proposed = weighted_mean(
            [perceptual_ev, subject_delta, salient_delta],
            [0.48, 0.34 * subject_weight, 0.18],
        )
    else:
        proposed = weighted_mean([perceptual_ev, salient_delta], [0.72, 0.28])

    # Do not brighten an already-highlight-heavy image simply because its
    # median is dark. Conversely, do not darken a genuinely underexposed image
    # merely because a small bright area exists.
    if proposed > 0.0:
        if high_clip > 0.25:
            proposed *= 0.35
        elif any_high_clip > 2.5 or bright_pct > 12.0:
            proposed *= 0.60
        if headroom < 0.45:
            proposed = min(proposed, max(0.0, headroom - 0.12))

    if proposed < 0.0:
        # Only reduce exposure aggressively when clipping is actually present.
        if high_clip < 0.05 and bright_pct < 3.0:
            proposed *= 0.80

    # Strong clipping should not generate a large global exposure move.
    if high_clip >= 1.0:
        proposed = min(proposed, 0.05)
    if low_clip >= 1.0 and proposed < 0:
        proposed *= 0.50

    proposed = clamp(proposed, -1.25, 1.0)

    # Suppress tiny corrections; these create visible changes without improving
    # the photograph.
    if abs(proposed) < 0.045:
        proposed = 0.0

    evidence_agreement = 1.0 - clamp(abs(perceptual_ev - salient_delta) / 1.2, 0.0, 1.0)
    confidence = clamp(
        0.42
        + 0.24 * evidence_agreement
        + 0.18 * subject_weight
        + 0.10 * float(scene.get("confidence", 0.45))
        - 0.16 * min(high_clip / 2.0, 1.0),
        0.25,
        0.95,
    )

    return Estimate(float(proposed), float(confidence))


def derive_edit_recommendations(
    tone: Mapping[str, Any],
    wb: Mapping[str, Any],
    contrast: Mapping[str, Any],
    detail: Mapping[str, Any],
    scene: Mapping[str, Any],
    subject_confidence: float,
) -> Dict[str, Any]:
    exposure = derive_exposure_recommendation(tone, scene, subject_confidence)

    clip = tone["clipping"]
    perceptual = tone["perceptual"]

    high_clip = float(clip["highlight_any_channel_percentage"])
    all_high_clip = float(clip["highlight_all_channels_percentage"])
    low_clip = float(clip["shadow_any_channel_percentage"])
    all_low_clip = float(clip["shadow_all_channels_percentage"])
    bright_pct = float(perceptual["bright_pixels_percentage"])
    dark_pct = float(perceptual["dark_pixels_percentage"])
    p10 = float(perceptual["percentiles_lab"].get("p10", 0.0))
    p25 = float(perceptual["percentiles_lab"].get("p25", 0.0))
    p75 = float(perceptual["percentiles_lab"].get("p75", 0.0))
    p90 = float(perceptual["percentiles_lab"].get("p90", 0.0))
    p95 = float(perceptual["percentiles_lab"].get("p95", 0.0))

    # Highlights/shadows are recovery recommendations, not substitutes for
    # exposure. They activate only when the distribution shows a real problem.
    highlights = 0.0
    if all_high_clip > 0.02:
        highlights = -clamp(all_high_clip * 8.0 + max(p95 - 225.0, 0.0) * 0.55, 0.0, 55.0)
    elif high_clip > 0.40 or bright_pct > 10.0:
        highlights = -clamp((high_clip - 0.40) * 2.0 + max(p95 - 238.0, 0.0) * 0.30, 0.0, 22.0)

    shadows = 0.0
    if all_low_clip > 0.02:
        shadows = clamp(all_low_clip * 7.0 + max(18.0 - p10, 0.0) * 0.30, 0.0, 45.0)
    elif low_clip > 0.40 or dark_pct > 18.0:
        shadows = clamp((low_clip - 0.40) * 1.8 + max(22.0 - p10, 0.0) * 0.22, 0.0, 25.0)

    # Contrast is based on actual percentile spread. Avoid forcing a fixed LAB
    # standard deviation, which can flatten high-key images or over-crunch low-key
    # images.
    dynamic_range = max(0.0, p75 - p25)
    contrast_rec = 0.0
    if dynamic_range < 24.0 and p10 > 12.0 and p90 < 242.0:
        contrast_rec = clamp((24.0 - dynamic_range) * 0.75, 0.0, 16.0)
    elif dynamic_range > 72.0 and (all_high_clip > 0.10 or all_low_clip > 0.10):
        contrast_rec = -clamp((dynamic_range - 72.0) * 0.35, 0.0, 10.0)

    noise_sigma = float(detail["noise_sigma_flat_regions"]["value"])
    noise_conf = float(detail["noise_sigma_flat_regions"]["confidence"])
    noise_reduction = clamp((noise_sigma - 2.0) * 3.0, 0.0, 30.0) * noise_conf

    subject_grad = float(detail["subject_gradient_mean"])
    sharp_conf = float(detail["sharpness_confidence"])
    sharpen = clamp((16.0 - subject_grad) * 1.15, 0.0, 18.0) * sharp_conf

    temp = float(wb["estimated_temperature_cast"]["value"])
    tint = float(wb["estimated_tint_cast"]["value"])
    wb_conf = float(wb["estimated_temperature_cast"]["confidence"])
    agreement = float(wb.get("estimator_agreement", 0.0))

    # Temperature/tint corrections are intentionally much smaller when the WB
    # estimators disagree. This prevents neutralizing intentional colored light.
    wb_strength = clamp(wb_conf * (0.55 + 0.45 * agreement), 0.0, 1.0)
    temperature_adjust = clamp(-temp * 1.15 * wb_strength, -18.0, 18.0)
    tint_adjust = clamp(-tint * 1.15 * wb_strength, -18.0, 18.0)
    if abs(temp) < 1.25:
        temperature_adjust = 0.0
    if abs(tint) < 1.25:
        tint_adjust = 0.0

    recommendation_confidence = clamp(
        weighted_mean(
            [exposure.confidence, wb_conf, noise_conf, sharp_conf, float(scene.get("confidence", 0.5))],
            [0.40, 0.22, 0.08, 0.08, 0.22],
        ),
        0.0,
        1.0,
    )

    return {
        "confidence": round(recommendation_confidence, 4),
        "exposure_ev": exposure.to_json(),
        "highlights": round(highlights, 3),
        "shadows": round(shadows, 3),
        "contrast": round(contrast_rec, 3),
        "temperature": {"value": round(temperature_adjust, 3), "confidence": round(wb_conf, 4)},
        "tint": {"value": round(tint_adjust, 3), "confidence": round(wb_conf, 4)},
        "noise_reduction": {"value": round(noise_reduction, 3), "confidence": round(noise_conf, 4)},
        "sharpen": {"value": round(sharpen, 3), "confidence": round(sharp_conf, 4)},
        "notes": [
            "Exposure uses adaptive percentile, subject and saliency evidence rather than a fixed scene luminance target.",
            "Scene classification is weak context and cannot by itself force exposure.",
            "White balance correction is confidence- and estimator-agreement-weighted.",
            "Highlights and shadows are recommended only when clipping or distribution evidence supports recovery.",
            "Contrast is based on percentile spread instead of a fixed LAB standard deviation target.",
        ],
    }


