from __future__ import annotations

from .core import *
from .io import *
from .analysis import *
from .color import *
from .effects import *

def integrate_mask_with_surrounding(
    roi: np.ndarray,
    working: np.ndarray,
    target_stats: Dict[str, float],
    surrounding_stats: Dict[str, float],
    strength: float,
) -> np.ndarray:
    """Blend a local mask back toward the surrounding scene so it feels integrated."""

    ring_mean = float(surrounding_stats.get("brightness", 0.5))
    roi_mean = float(target_stats.get("brightness", ring_mean))
    working_gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY).astype(np.float32)
    working_mean = float(np.mean(working_gray) / 255.0)

    target_mean = np.clip(
        0.55 * ring_mean + 0.45 * roi_mean,
        0.10,
        0.90,
    )

    delta = target_mean - working_mean
    if abs(delta) < 0.02:
        return working

    amount = float(np.clip(0.25 + 0.45 * strength, 0.18, 0.70))

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L = np.clip(L + delta * 255.0 * amount, 0.0, 255.0)
    lab[:, :, 0] = L

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================================================
# MASK-SPECIFIC ENHANCEMENT
# ============================================================================

def enhance_region(
    image: np.ndarray,
    analysis: ImageAnalysis,
    mask: np.ndarray,
    surrounding: np.ndarray,
    target_stats: Dict[str, float],
    surrounding_stats: Dict[str, float],
    name: str,
    global_strength: float,
    global_gains: Tuple[float, float, float],
    recommendation: Optional[Dict[str, Any]] = None,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:

    profile = MASK_PROFILES.get(
        name.lower(),
        DEFAULT_PROFILE,
    )

    pixels = target_stats["pixels"]

    if pixels < CONFIG.minimum_mask_pixels:

        return image.copy(), {
            "name": name,
            "pixels": pixels,
            "skipped": True,
            "reason": "too_few_pixels",
            "analysis": target_stats,
        }

    # ------------------------------------------------------------------------
    # Effective strength.
    # ------------------------------------------------------------------------

    strength = (
        profile.strength
        * global_strength
    )

    if pixels < 1500:
        strength *= 0.82

    elif pixels < 5000:
        strength *= 0.92

    strength = float(
        np.clip(
            strength,
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------------------------
    # Bounding box.
    # ------------------------------------------------------------------------

    bbox = mask_bbox(
        mask,
        padding=CONFIG.ring_outer_radius + 4,
    )

    if bbox is None:

        return image.copy(), {
            "name": name,
            "pixels": pixels,
            "skipped": True,
            "reason": "empty_mask",
            "analysis": target_stats,
        }

    x1, y1, x2, y2 = bbox

    roi = image[
        y1:y2,
        x1:x2,
    ]

    mask_roi = mask[
        y1:y2,
        x1:x2,
    ]

    # ------------------------------------------------------------------------
    # Local analysis.
    # ------------------------------------------------------------------------

    roi_analysis = analysis.crop(
        y1,
        y2,
        x1,
        x2,
    )

    target_roi_stats = analyze_region(
        roi_analysis,
        mask_roi,
    )

    ring_roi = surrounding[
        y1:y2,
        x1:x2,
    ]

    ring_stats = analyze_region(
        roi_analysis,
        ring_roi,
    )

    # ------------------------------------------------------------------------
    # Local neutral illumination.
    # ------------------------------------------------------------------------

    local_gains, wb_confidence = (
        estimate_local_neutral_gains(
            roi_analysis,
            ring_roi,
            global_gains,
        )
    )

    working = roi.copy()

    # ------------------------------------------------------------------------
    # White balance.
    # ------------------------------------------------------------------------

    working = apply_neutral_color(
        working,
        local_gains,
        wb_confidence,
        profile,
    )

    # ------------------------------------------------------------------------
    # Very subtle LAB cast correction.
    # ------------------------------------------------------------------------

    working = (
        apply_subtle_lab_neutralization(
            working,
            target_roi_stats,
            ring_stats,
            wb_confidence,
            profile,
        )
    )

    # ------------------------------------------------------------------------
    # Exposure.
    # ------------------------------------------------------------------------

    exposure = calculate_exposure(
        target_roi_stats,
        ring_stats,
        profile,
    )

    planned = recommendation.get("adjustments", {}) if recommendation else {}
    plan_match = float(
        np.clip(
            recommendation.get("context_match", {}).get("score", 1.0)
            if recommendation else 1.0,
            0.0,
            1.0,
        )
    )
    if isinstance(planned, dict):
        exposure = float(np.clip(
            exposure * 0.75 + float(planned.get("exposure", 0.0)) * 32.0 * 0.25,
            -CONFIG.max_exposure_change * 0.45,
            CONFIG.max_exposure_change,
        ))

    # ------------------------------------------------------------------------
    # Shadow / highlight recovery.
    # ------------------------------------------------------------------------

    shadow_lift = calculate_shadow_lift(
        target_roi_stats,
        profile,
    )

    highlight_reduction = (
        calculate_highlight_reduction(
            target_roi_stats,
            profile,
        )
    )

    if isinstance(planned, dict):
        shadow_lift = float(np.clip(
            shadow_lift * 0.75 + float(planned.get("shadows", 0.0)) * 0.35 * 0.25,
            0.0,
            CONFIG.max_shadow_lift,
        ))
        highlight_reduction = float(np.clip(
            highlight_reduction * 0.75 + abs(float(planned.get("highlights", 0.0))) * 0.35 * 0.25,
            0.0,
            CONFIG.max_highlight_reduction,
        ))

    # ------------------------------------------------------------------------
    # Adaptive contrast.
    # ------------------------------------------------------------------------

    target_contrast = (
        target_roi_stats["contrast"]
        * 255.0
    )

    surrounding_contrast = (
        ring_stats["contrast"]
        * 255.0
    )

    contrast_difference = (
        surrounding_contrast
        - target_contrast
    )

    contrast_amount = 0.0

    if abs(contrast_difference) > 5:

        contrast_amount = (
            contrast_difference
            / 255.0
            * CONFIG.contrast_strength
            * profile.contrast
        )

        contrast_amount = float(
            np.clip(
                contrast_amount,
                -0.10,
                0.14,
            )
        )

    if isinstance(planned, dict):
        contrast_amount = float(np.clip(
            contrast_amount * 0.75 + float(planned.get("contrast", 0.0)) / 100.0 * 0.25,
            -0.10,
            0.14,
        ))

    # A low extraction-time boundary score further reduces the live plan.
    strength *= 0.45 + 0.55 * plan_match

    # ------------------------------------------------------------------------
    # Tone.
    # ------------------------------------------------------------------------

    working = apply_tonal_adjustment(
        working,
        exposure,
        shadow_lift,
        highlight_reduction,
        contrast_amount,
        profile,
    )

    # ------------------------------------------------------------------------
    # Clarity.
    # ------------------------------------------------------------------------

    working = apply_local_clarity(
        working,
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Vibrance.
    # ------------------------------------------------------------------------

    working = apply_vibrance(
        working,
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Sharpening.
    # ------------------------------------------------------------------------

    working = apply_luminance_sharpening(
        working,
        target_roi_stats["sharpness"],
        strength,
        profile,
    )

    # ------------------------------------------------------------------------
    # Integrate the mask back into the surrounding scene so it does not feel
    # isolated, too dark, or too bright. This is the final step that keeps the
    # local correction natural and image-consistent.
    # ------------------------------------------------------------------------

    working = integrate_mask_with_surrounding(
        roi,
        working,
        target_roi_stats,
        ring_stats,
        strength,
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # No blending happens here.
    #
    # The compositor is the ONLY place that blends the mask.
    # ------------------------------------------------------------------------

    enhanced_full = image.copy()

    # Keep processed pixels inside the semantic mask. The padded ROI is only
    # context for local filters; writing it back in full creates rectangular
    # seams at the ROI boundary.
    mask_coverage = mask_roi > 0.01
    enhanced_roi = np.where(
        mask_coverage[:, :, None],
        working,
        roi,
    )

    enhanced_full[
        y1:y2,
        x1:x2,
    ] = enhanced_roi

    report = {
        "name": name,
        "pixels": pixels,
        "strength": strength,
        "skipped": False,

        "bbox": [
            x1,
            y1,
            x2,
            y2,
        ],

        "analysis": target_stats,
        "surrounding": surrounding_stats,
        "context_match": (
            recommendation.get("context_match")
            if recommendation else None
        ),

        "corrections": {
            "exposure": float(exposure),
            "shadow_lift": float(shadow_lift),
            "highlight_reduction": float(
                highlight_reduction
            ),
            "contrast": float(
                contrast_amount
            ),
            "wb_confidence": float(
                wb_confidence
            ),
            "local_gains": [
                float(local_gains[0]),
                float(local_gains[1]),
                float(local_gains[2]),
            ],
        },

        "profile": asdict(
            profile
        ),
    }

    return (
        enhanced_full,
        report,
    )


