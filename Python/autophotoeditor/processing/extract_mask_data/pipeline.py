from __future__ import annotations

from .shared import *
from .masks import *
from .analysis import *
from .profiles import *
from .enhancement import *

def _extract_grabcut_fallback(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Provide a deterministic foreground mask when the model is unavailable."""
    height, width = image.shape[:2]
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    margin_x = max(1, width // 12)
    margin_y = max(1, height // 12)
    mask[margin_y:height - margin_y, margin_x:width - margin_x] = cv2.GC_PR_FGD
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, background, foreground, 3, cv2.GC_INIT_WITH_MASK)
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1.0,
        0.0,
    ).astype(np.float32)
    return {"foreground": result}


def run_pipeline(
    input_path: Path,
    output_json: Path,
    person_model: str,
    human_model: str = "",
    face_model: str = "",
    device: str = "auto",
    person_imgsz: int = 1280,
    person_conf: float = 0.25,
    person_iou: float = 0.45,
    face_threshold: float = 0.5,
    face_padding: float = 0.10,
    visual_path: Optional[Path] = None,
    visual_dir: Optional[Path] = None,
    binary_dir: Optional[Path] = None,
    soft_dir: Optional[Path] = None,
    person_masks_dir: Optional[Path] = None,
    mask_opacity: float = 0.6,
    jpeg_quality: int = 95,
    pretty: bool = True,
    no_rle: bool = False,
    save_person_previews: bool = False,
    feather_sigma: float = 1.5,
) -> None:
    """Extract masks and immediately create a per-mask enhancement plan."""
    del human_model, face_model, person_imgsz, person_conf, person_iou
    del face_threshold, face_padding, visual_path, visual_dir, soft_dir
    del person_masks_dir, mask_opacity, jpeg_quality, no_rle
    del save_person_previews, feather_sigma

    image = decode_color(input_path)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Could not decode a 3-channel image: {input_path}")

    try:
        masks = _extract_model_masks(image, person_model, device)
    except Exception as exc:
        logger.warning("Model mask extraction failed, using GrabCut: %s", exc)
        masks = _extract_grabcut_fallback(image)

    if not masks:
        raise RuntimeError("No usable masks were extracted.")

    mask_directory = binary_dir or output_json.parent / "binary_masks"
    mask_entries: Dict[str, Dict[str, Any]] = {}
    for name, mask in masks.items():
        file_name = f"{name}.png"
        _write_mask(mask_directory / file_name, mask)
        mask_entries[name] = {
            "binary_file": str(Path(mask_directory.name) / file_name),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }

    mask_data = {
        "version": "2.0",
        "type": "semantic_masks",
        "image": {
            "path": str(input_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "masks": mask_entries,
    }
    save_json(mask_data, output_json, pretty)

    enhancement_path = output_json.with_name(
        f"{output_json.stem}_enhancements.json"
    )
    run(input_path, output_json, enhancement_path, strength=1.0, pretty=pretty)
    with enhancement_path.open("r", encoding="utf-8") as handle:
        enhancement_data = json.load(handle)

    mask_data["enhancement_json"] = str(enhancement_path)
    recommendations = enhancement_data.get("masks", {})
    for name, entry in mask_entries.items():
        if name in recommendations:
            entry["enhancement"] = recommendations[name]
    save_json(mask_data, output_json, pretty)


# ============================================================
# ATOMIC JSON
# ============================================================

def save_json(
    data: Dict[str, Any],
    path: Path,
    pretty: bool,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )

    temporary.replace(
        path
    )


# ============================================================
# OPTIONAL DEBUG REPORT
# ============================================================

def print_summary(
    results: Dict[str, Dict[str, Any]],
) -> None:

    print()
    print("=" * 96)
    print(
        f"{'MASK':<18}"
        f"{'EV':>8}"
        f"{'CON':>8}"
        f"{'HI':>8}"
        f"{'SH':>8}"
        f"{'WB':>8}"
        f"{'TINT':>8}"
        f"{'SAT':>8}"
        f"{'CLAR':>8}"
    )
    print("=" * 96)

    for name, result in results.items():

        if not result.get(
            "enabled",
            False,
        ):
            continue

        a = result[
            "adjustments"
        ]

        wb = a[
            "temperature"
        ]

        print(
            f"{name:<18}"
            f"{a['exposure']:>8.2f}"
            f"{a['contrast']:>8.1f}"
            f"{a['highlights']:>8.1f}"
            f"{a['shadows']:>8.1f}"
            f"{wb:>8.1f}"
            f"{a['tint']:>8.1f}"
            f"{a['saturation']:>8.1f}"
            f"{a['clarity']:>8.1f}"
        )

    print("=" * 96)
    print()


# ============================================================
# PIPELINE
# ============================================================

def run(
    input_path: Path,
    masks_json: Path,
    output_json: Path,
    strength: float,
    pretty: bool,
) -> None:

    if not input_path.exists():
        fail(
            f"Input image does not exist:\n{input_path}"
        )

    if not masks_json.exists():
        fail(
            f"Mask JSON does not exist:\n{masks_json}"
        )

    logger.info(
        "Loading image: %s",
        input_path,
    )

    image = decode_color(
        input_path
    )

    if image is None:
        fail(
            "Could not decode image."
        )

    if image.ndim != 3:
        fail(
            "Expected 3-channel color image."
        )

    height, width = image.shape[:2]

    logger.info(
        "Image: %dx%d",
        width,
        height,
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    logger.info(
        "Loading masks: %s",
        masks_json,
    )

    hard_masks, soft_masks = (
        load_masks_from_json(
            masks_json,
            (height, width),
        )
    )

    logger.info(
        "Loaded %d hard masks.",
        len(hard_masks),
    )

    logger.info(
        "Loaded %d soft masks.",
        len(soft_masks),
    )

    if not hard_masks and not soft_masks:
        fail(
            "No usable masks were found in the mask JSON."
        )

    # --------------------------------------------------------
    # Analyzer
    # --------------------------------------------------------

    enhancer = AutoMaskEnhancer(
        image=image,
        hard_masks=hard_masks,
        soft_masks=soft_masks,
        global_strength=strength,
    )

    # --------------------------------------------------------
    # Global correction
    # --------------------------------------------------------

    logger.info(
        "Calculating global auto enhancement..."
    )

    global_values = (
        calculate_global_enhancement(
            enhancer.data,
            enhancer.global_stats,
        )
    )

    # --------------------------------------------------------
    # Per-mask correction
    # --------------------------------------------------------

    logger.info(
        "Calculating per-mask enhancement..."
    )

    results = enhancer.analyze()

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    logger.info(
        "Running mask consistency pass..."
    )

    smooth_mask_values(
        results
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    output = build_output(
        input_path=input_path,
        masks_json=masks_json,
        image=image,
        enhancer=enhancer,
        mask_results=results,
        global_values=global_values,
    )

    save_json(
        output,
        output_json,
        pretty,
    )

    logger.info(
        "Enhancement JSON saved: %s",
        output_json,
    )

    print_summary(
        results
    )


