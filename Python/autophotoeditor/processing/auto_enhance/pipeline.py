from __future__ import annotations

from .core import *
from .analysis import *
from .recommendations import *

def classify_image(
    tone: Mapping[str, Any],
    wb: Mapping[str, Any],
    detail: Mapping[str, Any],
    scene: Mapping[str, Any],
    recommendations: Mapping[str, Any],
) -> Dict[str, Any]:
    exp = float(recommendations["exposure_ev"]["value"])
    exp_conf = float(recommendations["exposure_ev"]["confidence"])

    if exp > 0.55:
        exposure_status = "underexposed"
    elif exp > 0.18:
        exposure_status = "slightly_underexposed"
    elif exp < -0.55:
        exposure_status = "overexposed"
    elif exp < -0.18:
        exposure_status = "slightly_overexposed"
    else:
        exposure_status = "balanced_for_detected_scene"

    clip = tone["clipping"]
    hclip = float(clip["highlight_all_channels_percentage"])
    sclip = float(clip["shadow_all_channels_percentage"])

    highlight_status = "severe_clipping" if hclip > 2.0 else "clipping" if hclip > 0.5 else "normal"
    shadow_status = "severe_clipping" if sclip > 2.0 else "clipping" if sclip > 0.5 else "normal"

    noise = float(detail["noise_sigma_flat_regions"]["value"])
    noise_status = "very_low" if noise < 1.5 else "low" if noise < 3.0 else "moderate" if noise < 6.0 else "high"

    grad = float(detail["subject_gradient_mean"])
    sharpness_status = "very_soft" if grad < 5 else "soft" if grad < 10 else "normal" if grad < 24 else "high_detail"

    return {
        "scene": scene,
        "exposure_status": exposure_status,
        "exposure": {
            "status": exposure_status,
            "confidence": round(exp_conf, 4),
        },
        "highlight_status": highlight_status,
        "shadow_status": shadow_status,
        "sharpness_status": sharpness_status,
        "noise_status": noise_status,
        "color_cast": wb["interpretation"],
    }


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_image_data(
    image_path: str,
    analysis_max_dimension: int = DEFAULT_ANALYSIS_MAX_DIMENSION,
    mask_json_path: Optional[str] = None,
    enable_ai: bool = False,
    ai_model: str = "openai/clip-vit-base-patch32",
) -> Dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    try:
        img = decode_color(os.path.abspath(str(path)))
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc

    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Image decoder returned an invalid image.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {img.shape}.")
    if img.dtype != np.uint8:
        # Keep behavior predictable for this analyzer. RAW/16-bit should be handled
        # by a dedicated linear pipeline rather than silently cast here.
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w, c = img.shape
    file_size = path.stat().st_size
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    analysis_img = resize_longest(img, analysis_max_dimension)
    ah, aw = analysis_img.shape[:2]

    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)

    rgb = bgr8_to_rgb_float(analysis_img)
    rgb_linear = srgb_to_linear(rgb)
    linear_y = linear_luminance(rgb_linear)

    external_masks = load_external_masks(mask_json_path, (ah, aw))
    semantic_masks = estimate_semantic_color_masks(analysis_img, hsv, lab)
    subject_mask, saliency_map, subject_method, subject_confidence = derive_subject_mask(
        analysis_img,
        external_masks,
    )

    tone = analyze_tone(analysis_img, lab, linear_y, subject_mask, saliency_map)
    wb = analyze_white_balance(
        analysis_img,
        hsv,
        lab,
        rgb_linear,
        semantic_masks,
        external_masks,
    )
    color_profile = analyze_color_profile(analysis_img, hsv, wb)
    contrast = analyze_contrast(lab[:, :, 0], linear_y)
    detail = analyze_detail(gray, subject_mask)
    entropy = calculate_entropy(gray)

    scene = classify_scene(semantic_masks, external_masks, subject_mask, tone)

    semantic_regions: Dict[str, Any] = {
        "subject": analyze_mask_region("subject", subject_mask, lab, hsv, linear_y)
    }

    # Prefer externally supplied masks when available, otherwise use conservative heuristics.
    region_names = sorted(set(semantic_masks.keys()) | set(external_masks.keys()))
    for name in region_names:
        if name in ("neutral_candidates", "warm_pixels"):
            continue
        mask = external_masks.get(name, semantic_masks.get(name))
        if mask is not None:
            semantic_regions[name] = analyze_mask_region(name, ensure_mask(mask, (ah, aw)), lab, hsv, linear_y)

    grid_regions = analyze_grid_regions(analysis_img, rows=3, cols=3)

    recommendations = derive_edit_recommendations(
        tone,
        wb,
        contrast,
        detail,
        scene,
        subject_confidence,
    )

    classification = classify_image(tone, wb, detail, scene, recommendations)

    ai = OptionalAIAnalyzer(enable_ai, ai_model)
    ai_result = ai.analyze(analysis_img)

    histograms = {
        "blue_256bins_normalized": normalized_histogram(analysis_img[:, :, 0]),
        "green_256bins_normalized": normalized_histogram(analysis_img[:, :, 1]),
        "red_256bins_normalized": normalized_histogram(analysis_img[:, :, 2]),
        "luminance_lab_256bins_normalized": normalized_histogram(lab[:, :, 0]),
        "saturation_256bins_normalized": normalized_histogram(hsv[:, :, 1]),
        "linear_luminance_128bins_normalized": normalized_histogram(linear_y, bins=128, value_range=(0.0, 1.0)),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "width": int(w),
            "height": int(h),
            "channels": int(c),
            "aspect_ratio": round(aspect_ratio, 6),
            "total_pixels": int(w * h),
            "file_size_bytes": int(file_size),
            "file_path": str(path.resolve()),
            "analysis_width": int(aw),
            "analysis_height": int(ah),
            "analysis_max_dimension": int(analysis_max_dimension),
        },
        "analysis_method": {
            "photometric": "sRGB linear-light + perceptual LAB/HSV",
            "subject_method": subject_method,
            "subject_confidence": round(subject_confidence, 4),
            "external_masks_loaded": sorted(external_masks.keys()),
            "ai_requested": bool(enable_ai),
        },
        "exposure_and_tone": tone,
        "white_balance": wb,
        "contrast": contrast,
        "color_profile": color_profile,
        "quality": {
            **detail,
            # v2 compatibility alias; v3 value is estimated on flat regions.
            "noise_estimate_std": round(float(detail["noise_sigma_flat_regions"]["value"]), 4),
            "entropy": round(entropy, 4),
        },
        "scene": scene,
        "regions": {
            "semantic": semantic_regions,
            "grid_3x3": grid_regions,
        },
        "histograms": histograms,
        "ai": ai_result,
        "recommendations": recommendations,
        "classification": classification,
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data: Dict[str, Any], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid photographic image analyzer with confidence-aware recommendations."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_json", type=str, help="Path to save the JSON analysis.")
    parser.add_argument(
        "--analysis-size",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_DIMENSION,
        help=f"Maximum dimension used for analysis. Default: {DEFAULT_ANALYSIS_MAX_DIMENSION}.",
    )
    parser.add_argument(
        "--masks-json",
        type=str,
        default=None,
        help="Optional JSON containing paths to semantic mask images (person/face/skin/sky/etc.).",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Enable optional local zero-shot CLIP scene classification via transformers.",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Transformers model used by --ai.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a concise human-readable summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.analysis_size < 256:
        parser.error("--analysis-size must be at least 256.")

    try:
        data = extract_image_data(
            args.input_image,
            analysis_max_dimension=args.analysis_size,
            mask_json_path=args.masks_json,
            enable_ai=args.ai,
            ai_model=args.ai_model,
        )
        save_json(data, args.output_json)

        if args.pretty:
            rec = data["recommendations"]
            scene = data["scene"]
            wb = data["white_balance"]
            quality = data["quality"]
            classification = data["classification"]

            print("=" * 72)
            print("AUTO PHOTO EDITOR - HYBRID IMAGE ANALYSIS V3")
            print("=" * 72)
            print(f"Input              : {args.input_image}")
            print(f"Output             : {args.output_json}")
            print(f"Scene              : {scene['type']} ({scene['confidence']:.2f})")
            print(f"Exposure status    : {classification['exposure']['status']}")
            print(
                f"Exposure suggestion: {rec['exposure_ev']['value']:+.3f} EV "
                f"(confidence {rec['exposure_ev']['confidence']:.2f})"
            )
            print(
                f"WB cast            : temp {wb['estimated_temperature_cast']['value']:+.3f}, "
                f"tint {wb['estimated_tint_cast']['value']:+.3f} "
                f"(confidence {wb['estimated_temperature_cast']['confidence']:.2f})"
            )
            print(
                f"Noise sigma        : {quality['noise_sigma_flat_regions']['value']:.3f} "
                f"(confidence {quality['noise_sigma_flat_regions']['confidence']:.2f})"
            )
            print(f"Subject method     : {data['analysis_method']['subject_method']}")
            print(f"AI available       : {data['ai'].get('available', False)}")
            print("-" * 72)
            print("Suggested edits")
            print(f"  Exposure         : {rec['exposure_ev']['value']:+.3f} EV")
            print(f"  Highlights       : {rec['highlights']:+.1f}")
            print(f"  Shadows          : {rec['shadows']:+.1f}")
            print(f"  Contrast         : {rec['contrast']:+.1f}")
            print(f"  Temperature      : {rec['temperature']['value']:+.1f}")
            print(f"  Tint             : {rec['tint']['value']:+.1f}")
            print(f"  Noise reduction  : {rec['noise_reduction']['value']:+.1f}")
            print(f"  Sharpen          : {rec['sharpen']['value']:+.1f}")
            print("=" * 72)
        else:
            print(f"Saved hybrid image analysis to: {args.output_json}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Image analysis failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
