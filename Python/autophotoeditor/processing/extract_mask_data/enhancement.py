from __future__ import annotations

from .shared import *
from .masks import *
from .analysis import *
from .profiles import *

def calculate_global_enhancement(
    data: ImageColorData,
    stats: GlobalSceneStats,
) -> EnhancementValues:

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    median = max(
        stats.p50,
        0.001,
    )

    exposure = math.log2(
        0.18 / median
    )

    # Prevent lifting already bright photographs.
    if stats.p95 > 0.92:
        exposure *= 0.50

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    highlights = 0.0

    if stats.p95 > 0.94:
        highlights = -clip(
            (
                stats.p95
                -
                0.94
            )
            * 220.0,
            0.0,
            35.0,
        )

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    shadows = 0.0

    if stats.p10 < 0.035:

        shadows = clip(
            (
                0.035
                -
                stats.p10
            )
            * 350.0,
            0.0,
            35.0,
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    dynamic_range = (
        stats.p95
        -
        stats.p05
    )

    contrast = clip(
        (
            dynamic_range
            -
            0.65
        )
        * 55.0,
        -20.0,
        20.0,
    )

    # --------------------------------------------------------
    # Whites / blacks
    # --------------------------------------------------------

    whites = 0.0
    blacks = 0.0

    if stats.p99 > 0.985:
        whites = -clip(
            (
                stats.p99
                -
                0.985
            )
            * 180.0,
            0.0,
            20.0,
        )

    if stats.p01 < 0.008:
        blacks = clip(
            (
                0.008
                -
                stats.p01
            )
            * 600.0,
            0.0,
            12.0,
        )

    return limit_values(
        EnhancementValues(
            exposure=clip(
                exposure,
                -1.0,
                1.0,
            ),
            contrast=contrast,
            highlights=highlights,
            shadows=shadows,
            whites=whites,
            blacks=blacks,
        )
    )


# ============================================================
# CONSISTENCY PASS
# ============================================================

def smooth_mask_values(
    masks: Dict[str, Dict[str, Any]],
) -> None:

    """
    Prevent neighboring / hierarchical masks from receiving
    wildly different corrections.

    This is especially important for:

        face
        skin
        skin_face
        skin_neck
        body

    because these masks overlap.
    """

    groups = [
        [
            "skin",
            "skin_face",
            "skin_neck",
        ],

        [
            "face",
            "skin",
        ],

        [
            "clothing",
            "top",
            "dress",
            "skirt",
            "pants",
            "torso",
        ],
    ]

    for group in groups:

        available = [
            name
            for name in group
            if name in masks
            and masks[name].get(
                "enabled",
                False,
            )
        ]

        if len(available) < 2:
            continue

        fields = [
            "exposure",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "saturation",
            "vibrance",
            "clarity",
            "dehaze",
        ]

        for field in fields:

            values = [
                masks[name][
                    "adjustments"
                ][field]
                for name in available
            ]

            median = float(
                np.median(values)
            )

            for name in available:

                current = masks[name][
                    "adjustments"
                ][field]

                # Move only 25% toward group median.
                corrected = (
                    current * 0.75
                    +
                    median * 0.25
                )

                low, high = LIMITS[
                    field
                ]

                masks[name][
                    "adjustments"
                ][field] = clip(
                    corrected,
                    low,
                    high,
                )


# ============================================================
# JSON OUTPUT
# ============================================================

def build_output(
    input_path: Path,
    masks_json: Path,
    image: np.ndarray,
    enhancer: AutoMaskEnhancer,
    mask_results: Dict[str, Dict[str, Any]],
    global_values: EnhancementValues,
) -> Dict[str, Any]:

    height, width = image.shape[:2]

    return {
        "version": "1.0",

        "type": "auto_mask_enhancement",

        "description": (
            "Color-science and OpenCV based automatic "
            "per-mask Lightroom-style enhancement analysis."
        ),

        "image": {
            "path": str(input_path),
            "width": width,
            "height": height,
            "channels": 3,
            "color_format": "BGR",
            "working_color_space": (
                "sRGB -> linear RGB -> CIE Lab"
            ),
            "luminance_model": (
                "Rec.709 linear-light luminance"
            ),
        },

        "source_masks": {
            "json": str(masks_json),
        },

        "global_analysis": {
            "luminance": {
                "p01": enhancer.global_stats.p01,
                "p05": enhancer.global_stats.p05,
                "p10": enhancer.global_stats.p10,
                "p25": enhancer.global_stats.p25,
                "p50": enhancer.global_stats.p50,
                "p75": enhancer.global_stats.p75,
                "p90": enhancer.global_stats.p90,
                "p95": enhancer.global_stats.p95,
                "p99": enhancer.global_stats.p99,
                "mean": enhancer.global_stats.mean,
                "rms": enhancer.global_stats.rms,
            },

            "clipping": {
                "black": (
                    enhancer.global_stats.clipping_black
                ),
                "white": (
                    enhancer.global_stats.clipping_white
                ),
            },
        },

        "global_enhancement": {
            "enabled": True,
            "adjustments": asdict(
                global_values
            ),
        },

        "masks": mask_results,

        "processing": {
            "algorithm": [
                "linear-light RGB conversion",
                "Rec.709 luminance analysis",
                "CIE Lab chromaticity analysis",
                "robust percentile exposure analysis",
                "highlight/shadow clipping analysis",
                "local surrounding-region analysis",
                "neutral white-balance estimation",
                "mask-specific correction",
                "hierarchical correction smoothing",
                "conservative adjustment limiting",
            ],

            "neutral_color_policy": (
                "Correct chromatic cast relative to the "
                "local surrounding region rather than "
                "forcing naturally colored objects to gray."
            ),

            "mask_policy": (
                "Each detected region receives independent "
                "adjustment values."
            ),
        },
    }


# ============================================================
# MASK EXTRACTION PIPELINE
# ============================================================

def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError(f"Could not encode mask: {path}")
    encoded.tofile(str(path))


def _extract_model_masks(
    image: np.ndarray,
    model_path: str,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract semantic masks with Ultralytics, keeping one file per instance."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for mask extraction."
        ) from exc

    model = YOLO(model_path)
    result = model.predict(
        source=image,
        device=device,
        verbose=False,
        retina_masks=True,
    )[0]

    if result.masks is None or result.boxes is None:
        return {}

    masks: Dict[str, np.ndarray] = {}
    names = result.names or {}
    mask_data = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for index, raw_mask in enumerate(mask_data):
        if index >= len(classes):
            break

        mask = cv2.resize(
            raw_mask.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = np.clip(mask, 0.0, 1.0)
        if np.count_nonzero(mask > 0.15) < 20:
            continue

        label = str(names.get(int(classes[index]), "region")).strip().lower()
        base_name = label or "region"
        name = base_name
        suffix = 2
        while name in masks:
            name = f"{base_name}_{suffix}"
            suffix += 1
        masks[name] = mask

    return masks

