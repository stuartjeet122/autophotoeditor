from __future__ import annotations

from .core import *
from .io import *
from .analysis import *
from .color import *
from .effects import *
from .region import *

def composite_mask(
    base: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    ownership: np.ndarray,
    strength: float,
    feather_radius: float,
) -> np.ndarray:

    alpha = np.clip(
        mask.astype(np.float32),
        0.0,
        1.0,
    )

    # Remove weak rectangular/background coverage from imperfect mask PNGs
    # before feathering. Feathering a low-alpha rectangle makes the artifact
    # visible as a translucent box around the intended subject.
    alpha[alpha < 0.05] = 0.0

    # Strength is applied exactly once.
    alpha *= float(
        np.clip(
            strength,
            0.0,
            1.0,
        )
    )

    # Feather exactly once.
    alpha = feather_mask(
        alpha,
        feather_radius,
    )

    # Prevent weaker overlapping masks from overwriting
    # stronger semantic regions.
    available = np.clip(
        1.0 - ownership,
        0.0,
        1.0,
    )

    alpha *= available

    alpha[
        alpha < 0.003
    ] = 0.0

    alpha3 = alpha[:, :, None]

    base_f = base.astype(
        np.float32,
        copy=False,
    )

    enhanced_f = enhanced.astype(
        np.float32,
        copy=False,
    )

    result = (
        base_f
        * (1.0 - alpha3)
        +
        enhanced_f
        * alpha3
    )

    ownership[:] = np.maximum(
        ownership,
        alpha,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# SORTING
# ============================================================================

def mask_sort_key(
    name: str,
) -> int:

    name = name.lower()

    try:
        return MASK_PRIORITY.index(
            name
        )

    except ValueError:
        return (
            len(MASK_PRIORITY)
            + 100
        )


# ============================================================================
# PERSON MASK SUPPORT
# ============================================================================

def discover_person_masks(
    person_masks_dir: Optional[Path],
    width: int,
    height: int,
) -> Dict[
    str,
    Dict[str, np.ndarray],
]:

    if person_masks_dir is None:
        return {}

    if not person_masks_dir.exists():
        return {}

    result: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    for person_root in sorted(
        person_masks_dir.glob(
            "person_*"
        )
    ):

        if not person_root.is_dir():
            continue

        person_id = (
            person_root.name
        )

        soft_dir = (
            person_root / "soft"
        )

        binary_dir = (
            person_root / "binary"
        )

        source_dir = (
            soft_dir
            if soft_dir.exists()
            else binary_dir
        )

        if not source_dir.exists():
            continue

        person_masks: Dict[
            str,
            np.ndarray,
        ] = {}

        for path in source_dir.glob(
            "*.png"
        ):

            name = path.stem.lower()

            mask = load_mask(
                path,
                width,
                height,
            )

            if mask is None:
                continue

            mask = clean_mask(
                mask
            )

            if (
                np.count_nonzero(
                    mask > 0.05
                )
                <
                CONFIG.minimum_mask_pixels
            ):
                continue

            person_masks[name] = mask

        if person_masks:
            result[
                person_id
            ] = person_masks

    return result


# ============================================================================
# PROCESS ONE SET OF MASKS
# ============================================================================

def process_mask_set(
    image: np.ndarray,
    analysis: ImageAnalysis,
    loaded_masks: Dict[str, np.ndarray],
    final_image: np.ndarray,
    ownership: np.ndarray,
    global_strength: float,
    feather_radius: float,
    global_gains: Tuple[float, float, float],
    reports: List[Dict[str, Any]],
    save_mask_results: Optional[Path],
    progress: Optional[
        Callable[[int, str], None]
    ] = None,
    progress_start: int = 25,
    progress_end: int = 90,
    enhancement_plans: Optional[Dict[str, Dict[str, Any]]] = None,
) -> np.ndarray:

    if not loaded_masks:
        return final_image

    # Union of all masks.
    all_masks = np.zeros_like(
        next(iter(loaded_masks.values())),
        dtype=np.float32,
    )

    for mask in loaded_masks.values():
        all_masks = np.maximum(
            all_masks,
            mask,
            out=all_masks,
        )

    ordered_names = sorted(
        loaded_masks.keys(),
        key=mask_sort_key,
    )

    total = len(
        ordered_names
    )

    for index, name in enumerate(
        ordered_names,
        start=1,
    ):

        if progress is not None:

            percent = (
                progress_start
                +
                int(
                    index
                    * (
                        progress_end
                        - progress_start
                    )
                    /
                    max(
                        total,
                        1,
                    )
                )
            )

            progress(
                percent,
                f"enhance-mask:{name}",
            )

        mask = loaded_masks[
            name
        ]

        pixels = int(
            np.count_nonzero(
                mask > 0.05
            )
        )

        logger.info(
            "[%d/%d] %s pixels=%d",
            index,
            total,
            name,
            pixels,
        )

        # --------------------------------------------------------------------
        # Context.
        # --------------------------------------------------------------------

        surrounding_stats, ring = (
            analyze_surrounding(
                analysis,
                mask,
                all_masks=all_masks,
            )
        )

        # --------------------------------------------------------------------
        # Target.
        # --------------------------------------------------------------------

        target_stats = analyze_region(
            analysis,
            mask,
        )

        # --------------------------------------------------------------------
        # Enhance ORIGINAL image.
        # --------------------------------------------------------------------

        enhanced, report = (
            enhance_region(
                image,
                analysis,
                mask,
                ring,
                target_stats,
                surrounding_stats,
                name,
                global_strength,
                global_gains,
                recommendation=(enhancement_plans or {}).get(name),
            )
        )

        reports.append(
            report
        )

        if report.get(
            "skipped",
            False,
        ):
            continue

        profile = MASK_PROFILES.get(
            name.lower(),
            DEFAULT_PROFILE,
        )

        effective_strength = float(
            np.clip(
                profile.strength
                * global_strength,
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------------
        # SINGLE COMPOSITE.
        # --------------------------------------------------------------------

        final_image = composite_mask(
            final_image,
            enhanced,
            mask,
            ownership,
            effective_strength,
            feather_radius,
        )

        # --------------------------------------------------------------------
        # Optional preview.
        # --------------------------------------------------------------------

        if save_mask_results is not None:

            preview = image.copy()

            preview_ownership = (
                np.zeros_like(
                    ownership
                )
            )

            preview = composite_mask(
                preview,
                enhanced,
                mask,
                preview_ownership,
                effective_strength,
                feather_radius,
            )

            preview_path = (
                save_mask_results
                /
                f"{name}_enhanced.png"
            )

            imwrite_unicode(
                preview_path,
                preview,
            )

    return final_image


# ============================================================================
# MAIN PROCESS
# ============================================================================

def process(
    image_path: Path,
    json_path: Path,
    output_path: Path,
    mask_dir: Optional[Path] = None,
    global_strength: float = 1.0,
    feather_radius: float = 2.0,
    save_mask_results: Optional[Path] = None,
    progress: Optional[
        Callable[[int, str], None]
    ] = None,
    person_masks_dir: Optional[Path] = None,
    mask_names: Optional[List[str]] = None,
) -> None:

    def report_progress(
        percent: int,
        stage: str,
    ) -> None:

        if progress is not None:
            progress(
                max(
                    0,
                    min(
                        100,
                        int(percent),
                    ),
                ),
                stage,
            )

    # ========================================================================
    # LOAD IMAGE
    # ========================================================================

    report_progress(
        5,
        "load-image",
    )

    logger.info(
        "Loading image: %s",
        image_path,
    )

    image = imread_unicode(
        image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape[:2]

    logger.info(
        "Image size: %dx%d",
        width,
        height,
    )

    # Immutable original.
    source = image.copy()

    # Final compositor.
    final_image = image.copy()

    # ========================================================================
    # JSON
    # ========================================================================

    report_progress(
        10,
        "load-json",
    )

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # ========================================================================
    # MASK DISCOVERY
    # ========================================================================

    report_progress(
        18,
        "discover-masks",
    )

    masks = extract_mask_entries(
        data,
        json_path,
        mask_dir,
    )

    enhancement_plans = {
        str(name).lower(): item.get("enhancement", {})
        for name, item in data.get("masks", {}).items()
        if isinstance(item, dict) and isinstance(item.get("enhancement"), dict)
    }

    # Explicit mask directory wins.
    if mask_dir is not None:

        directory_masks = (
            discover_masks_from_directory(
                mask_dir
            )
        )

        masks.update(
            directory_masks
        )

    # ========================================================================
    # VALIDATE
    # ========================================================================

    clean_masks: Dict[
        str,
        Path,
    ] = {}

    image_resolved = (
        image_path.resolve()
    )

    for name, path in masks.items():

        name = (
            str(name)
            .strip()
            .lower()
        )

        try:

            if (
                path.resolve()
                ==
                image_resolved
            ):
                continue

        except Exception:
            continue

        if path.suffix.lower() != ".png":
            continue

        if not path.exists():
            continue

        clean_masks[
            name
        ] = path

    masks = clean_masks

    if mask_names is not None:
        requested_names = {
            str(name).strip().lower()
            for name in mask_names
            if str(name).strip()
        }
        masks = {
            name: path
            for name, path in masks.items()
            if name in requested_names
        }

    if not masks:

        raise RuntimeError(
            "No valid PNG masks were found."
        )

    # ========================================================================
    # LOAD MASKS
    # ========================================================================

    report_progress(
        22,
        "load-masks",
    )

    loaded_masks: Dict[
        str,
        np.ndarray,
    ] = {}

    for name, path in sorted(
        masks.items(),
        key=lambda item: mask_sort_key(
            item[0]
        ),
    ):

        mask = load_mask(
            path,
            width,
            height,
        )

        if mask is None:
            continue

        mask = clean_mask(
            mask
        )

        pixel_count = int(
            np.count_nonzero(
                mask > 0.05
            )
        )

        if (
            pixel_count
            <
            CONFIG.minimum_mask_pixels
        ):
            continue

        loaded_masks[
            name
        ] = mask

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    report_progress(
        25,
        "analyze-image",
    )

    logger.info(
        "Building image analysis cache..."
    )

    analysis = ImageAnalysis(
        source
    )

    # ========================================================================
    # GLOBAL NEUTRAL ILLUMINANT
    # ========================================================================

    report_progress(
        27,
        "estimate-neutral-light",
    )

    global_gains = (
        estimate_global_neutral_gains(
            analysis
        )
    )

    logger.info(
        "Global neutral gains: "
        "R=%.4f G=%.4f B=%.4f",
        global_gains[0],
        global_gains[1],
        global_gains[2],
    )

    # ========================================================================
    # OWNERSHIP
    # ========================================================================

    ownership = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    reports: List[
        Dict[str, Any]
    ] = []

    if save_mask_results is not None:

        save_mask_results.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # GLOBAL MASK PROCESSING
    # ========================================================================

    if loaded_masks:

        final_image = process_mask_set(
            image=source,
            analysis=analysis,
            loaded_masks=loaded_masks,
            final_image=final_image,
            ownership=ownership,
            global_strength=float(
                np.clip(
                    global_strength,
                    0.0,
                    1.0,
                )
            ),
            feather_radius=max(
                0.0,
                float(
                    feather_radius
                ),
            ),
            global_gains=global_gains,
            reports=reports,
            save_mask_results=save_mask_results,
            progress=progress,
            progress_start=28,
            progress_end=88,
            enhancement_plans=enhancement_plans,
        )

    # ========================================================================
    # PERSON MASK SUPPORT
    # ========================================================================

    person_masks = (
        discover_person_masks(
            person_masks_dir,
            width,
            height,
        )
    )

    if person_masks:

        logger.info(
            "Found %d person mask groups.",
            len(person_masks),
        )

        person_root_reports = []

        for person_id, person_group in (
            person_masks.items()
        ):

            logger.info(
                "Processing %s",
                person_id,
            )

            before_count = len(
                reports
            )

            final_image = (
                process_mask_set(
                    image=source,
                    analysis=analysis,
                    loaded_masks=person_group,
                    final_image=final_image,
                    ownership=ownership,
                    global_strength=float(
                        np.clip(
                            global_strength,
                            0.0,
                            1.0,
                        )
                    ),
                    feather_radius=max(
                        0.0,
                        float(
                            feather_radius
                        ),
                    ),
                    global_gains=global_gains,
                    reports=reports,
                    save_mask_results=None,
                    progress=progress,
                    progress_start=40,
                    progress_end=88,
                    enhancement_plans=enhancement_plans,
                )
            )

            after_count = len(
                reports
            )

            person_root_reports.append(
                {
                    "person_id": person_id,
                    "reports": reports[
                        before_count:
                        after_count
                    ],
                }
            )

        # Keep report information.
        reports.append(
            {
                "person_groups": person_root_reports
            }
        )

    # ========================================================================
    # FINAL GLOBAL SAFETY
    # ========================================================================

    report_progress(
        92,
        "finalize",
    )

    # Avoid any accidental NaN/Inf.
    final_image = np.nan_to_num(
        final_image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    final_image = np.clip(
        final_image,
        0,
        255,
    ).astype(np.uint8)

    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================

    report_progress(
        95,
        "save-output",
    )

    logger.info(
        "Saving output: %s",
        output_path,
    )

    if not imwrite_unicode(
        output_path,
        final_image,
        quality=95,
    ):

        raise RuntimeError(
            f"Could not save output: {output_path}"
        )

    # ========================================================================
    # REPORT
    # ========================================================================

    report_progress(
        98,
        "save-report",
    )

    report_path = (
        output_path.parent
        /
        f"{output_path.stem}_mask_report.json"
    )

    report_data = {
        "engine":
            "humanmaskextractor_adaptive_neutral_v6",

        "input":
            str(image_path),

        "mask_json":
            str(json_path),

        "mask_dir":
            str(mask_dir)
            if mask_dir
            else None,

        "person_masks_dir":
            str(person_masks_dir)
            if person_masks_dir
            else None,

        "output":
            str(output_path),

        "image_width":
            width,

        "image_height":
            height,

        "global_strength":
            float(global_strength),

        "feather_radius":
            float(feather_radius),

        "global_neutral_gains": [
            float(global_gains[0]),
            float(global_gains[1]),
            float(global_gains[2]),
        ],

        "masks_processed":
            sorted(
                loaded_masks.keys(),
                key=mask_sort_key,
            ),

        "reports":
            reports,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Report saved: %s",
        report_path,
    )

    report_progress(
        100,
        "completed",
    )

    logger.info(
        "DONE: %s",
        output_path,
    )


