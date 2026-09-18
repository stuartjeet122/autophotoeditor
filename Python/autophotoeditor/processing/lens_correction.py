#!/usr/bin/env python3

"""
lens_correction.py

Lens correction engine for AutoPhotoEditor.

Features
--------

1. Automatic EXIF camera/lens detection
2. Lensfun database lookup
3. Lensfun distortion correction
4. Lensfun TCA correction
5. Lensfun vignetting correction
6. Manual OpenCV distortion correction
7. Manual distortion strength
8. Manual optical center
9. Manual focal scaling
10. Automatic crop
11. Keep-full-frame mode
12. Profile listing/search
13. Debug grid
14. JPEG / PNG / TIFF support

Examples
--------

Automatic correction from EXIF:

    python lens_correction.py input.jpg output.jpg


Specify camera/lens manually:

    python lens_correction.py input.jpg output.jpg \
        --camera "Canon EOS 5D Mark IV" \
        --camera-maker "Canon" \
        --lens-maker "Sigma" \
        --lens "Sigma 24-70mm F2.8 DG OS HSM"


Manual correction:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --k3 0.0


Manual correction strength:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --strength 0.75


Manual center:

    python lens_correction.py input.jpg output.jpg \
        --manual \
        --k1 -0.20 \
        --k2 0.05 \
        --center-x 0.50 \
        --center-y 0.50


Search database:

    python lens_correction.py \
        --search-lens "24-70"


List cameras:

    python lens_correction.py \
        --list-cameras "Canon"


List lenses:

    python lens_correction.py \
        --list-lenses "Sigma"


Debug:

    python lens_correction.py input.jpg output.jpg \
        --debug
"""


from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from autophotoeditor.core.image_io import RAW_EXTENSIONS, decode_color


# ============================================================================
# OPTIONAL LENSPY IMPORT
# ============================================================================

try:
    import lensfunpy
except ImportError:
    lensfunpy = None


# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-8


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImageMetadata:
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_make: Optional[str] = None
    lens_model: Optional[str] = None

    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    focus_distance: Optional[float] = None

    width: int = 0
    height: int = 0


@dataclass
class ManualDistortion:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    p1: float = 0.0
    p2: float = 0.0

    center_x: float = 0.5
    center_y: float = 0.5

    focal_scale: float = 1.0

    strength: float = 1.0


@dataclass
class LensCorrectionOptions:
    use_lensfun: bool = True

    distortion: bool = True
    tca: bool = False
    vignetting: bool = False

    manual: bool = False

    auto_crop: bool = True
    crop_factor: float = 1.0

    interpolation: int = cv2.INTER_LANCZOS4

    debug: bool = False
    debug_grid: bool = False


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def normalize_string(
    value: Optional[str],
) -> Optional[str]:

    if value is None:
        return None

    value = str(value)

    value = (
        value
        .replace("\x00", "")
        .strip()
    )

    if not value:
        return None

    return value


def finite_image(
    image: np.ndarray,
) -> np.ndarray:

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================================
# IMAGE IO
# ============================================================================

def load_image(
    path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if os.path.splitext(path)[1].lower() in RAW_EXTENSIONS:
        image = decode_color(Path(path))
    else:
        image = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    # Grayscale
    if image.ndim == 2:

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

        return (
            rgb.astype(np.float32) / 255.0,
            None,
        )

    # RGBA
    if image.shape[2] == 4:

        bgr = image[..., :3]

        alpha = (
            image[..., 3]
            .astype(np.float32)
            / 255.0
        )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        return (
            rgb.astype(np.float32) / 255.0,
            alpha,
        )

    # RGB
    bgr = image[..., :3]

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        rgb.astype(np.float32) / 255.0,
        None,
    )


def save_image(
    path: str,
    image: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    jpeg_quality: int = 97,
):

    image = finite_image(image)

    rgb8 = np.round(
        image * 255.0
    ).astype(np.uint8)

    if alpha is not None:

        alpha8 = np.round(
            np.clip(alpha, 0.0, 1.0)
            * 255.0
        ).astype(np.uint8)

        rgba = np.dstack(
            [rgb8, alpha8]
        )

        output = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGRA,
        )

    else:

        output = cv2.cvtColor(
            rgb8,
            cv2.COLOR_RGB2BGR,
        )

    extension = (
        os.path.splitext(path)[1]
        .lower()
    )

    params = []

    if extension in (".jpg", ".jpeg"):

        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(
                np.clip(
                    jpeg_quality,
                    1,
                    100,
                )
            ),
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ]

    elif extension == ".png":

        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    if not cv2.imwrite(
        path,
        output,
        params,
    ):
        raise RuntimeError(
            f"Failed to save: {path}"
        )


# ============================================================================
# EXIF
# ============================================================================

def read_exif(
    path: str,
) -> ImageMetadata:

    metadata = ImageMetadata()

    tags = {}
    try:
        with Image.open(path) as image:
            metadata.width = image.width
            metadata.height = image.height
            exif = image.getexif()
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                tags[name] = value
    except (OSError, ValueError):
        if os.path.splitext(path)[1].lower() not in RAW_EXTENSIONS:
            raise
        try:
            import rawpy

            with rawpy.imread(path) as raw:
                metadata.width = int(raw.sizes.width)
                metadata.height = int(raw.sizes.height)
                raw_lens = getattr(raw, "lens", None)
                metadata.lens_model = normalize_string(
                    getattr(raw_lens, "model", None)
                )
                metadata.lens_make = normalize_string(
                    getattr(raw_lens, "make", None)
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not read RAW dimensions or lens metadata: {path}: {exc}"
            ) from exc

    metadata.camera_make = normalize_string(
        tags.get("Make")
    )

    metadata.camera_model = normalize_string(
        tags.get("Model")
    )

    metadata.lens_make = normalize_string(
        tags.get("LensMake")
    )

    metadata.lens_model = normalize_string(
        tags.get("LensModel")
    )

    metadata.focal_length = (
        safe_float(
            tags.get("FocalLength")
        )
    )

    metadata.aperture = (
        safe_float(
            tags.get("FNumber")
        )
    )

    metadata.focus_distance = (
        safe_float(
            tags.get("SubjectDistance")
        )
    )

    return metadata


# ============================================================================
# DISPLAY METADATA
# ============================================================================

def print_metadata(
    metadata: ImageMetadata,
):

    print()
    print("=" * 72)
    print("IMAGE METADATA")
    print("=" * 72)

    print(
        "Camera make :",
        metadata.camera_make,
    )

    print(
        "Camera model:",
        metadata.camera_model,
    )

    print(
        "Lens make   :",
        metadata.lens_make,
    )

    print(
        "Lens model  :",
        metadata.lens_model,
    )

    print(
        "Focal length:",
        metadata.focal_length,
    )

    print(
        "Aperture    :",
        metadata.aperture,
    )

    print(
        "Distance    :",
        metadata.focus_distance,
    )

    print(
        "Resolution  :",
        f"{metadata.width} x {metadata.height}",
    )

    print("=" * 72)
    print()


# ============================================================================
# LENSFUN
# ============================================================================

def require_lensfun():

    if lensfunpy is None:

        raise RuntimeError(
            "lensfunpy is not installed.\n\n"
            "Install it with:\n"
            "pip install lensfunpy"
        )


def create_database():

    require_lensfun()

    return lensfunpy.Database()


# ============================================================================
# CAMERA SEARCH
# ============================================================================

def find_camera(
    db,
    maker: Optional[str],
    model: Optional[str],
):

    if not model:
        return None

    cameras = []

    try:

        cameras = db.find_cameras(
            maker or "",
            model,
        )

    except Exception:

        try:
            cameras = db.find_cameras(
                model
            )
        except Exception:
            cameras = []

    if not cameras:
        return None

    # Exact-ish match first.
    model_lower = model.lower()

    exact = [
        cam
        for cam in cameras
        if model_lower
        in str(cam.model).lower()
    ]

    if exact:
        return exact[0]

    return cameras[0]


# ============================================================================
# LENS SEARCH
# ============================================================================

def find_lenses(
    db,
    camera,
    maker: Optional[str],
    model: Optional[str],
):

    lenses = []

    try:

        lenses = db.find_lenses(
            camera,
            maker or "",
            model or "",
        )

    except Exception:

        try:

            lenses = db.find_lenses(
                camera,
            )

        except Exception:

            lenses = []

    return lenses


def choose_best_lens(
    lenses,
    requested_model: Optional[str],
):

    if not lenses:
        return None

    if not requested_model:
        return lenses[0]

    requested = requested_model.lower()

    # Exact match.
    for lens in lenses:

        name = str(
            getattr(
                lens,
                "model",
                "",
            )
        ).lower()

        if name == requested:
            return lens

    # Partial match.
    for lens in lenses:

        name = str(
            getattr(
                lens,
                "model",
                "",
            )
        ).lower()

        if requested in name:
            return lens

    return lenses[0]


# ============================================================================
# LENSFUN PROFILE
# ============================================================================

def find_lensfun_profile(
    metadata: ImageMetadata,
):

    db = create_database()

    camera = find_camera(
        db,
        metadata.camera_make,
        metadata.camera_model,
    )

    if camera is None:

        raise RuntimeError(
            "Could not find camera in Lensfun database.\n"
            f"Camera: {metadata.camera_make} "
            f"{metadata.camera_model}"
        )

    lenses = find_lenses(
        db,
        camera,
        metadata.lens_make,
        metadata.lens_model,
    )

    if not lenses:

        raise RuntimeError(
            "Could not find lens in Lensfun database.\n"
            f"Lens: {metadata.lens_make} "
            f"{metadata.lens_model}"
        )

    lens = choose_best_lens(
        lenses,
        metadata.lens_model,
    )

    return db, camera, lens


# ============================================================================
# LENS PROFILE INFORMATION
# ============================================================================

def print_lens_profile(
    camera,
    lens,
):

    print()
    print("=" * 72)
    print("LENFUN PROFILE")
    print("=" * 72)

    print(
        "Camera:",
        camera,
    )

    print(
        "Lens:",
        lens,
    )

    crop_factor = getattr(
        camera,
        "crop_factor",
        None,
    )

    print(
        "Camera crop factor:",
        crop_factor,
    )

    print("=" * 72)
    print()


# ============================================================================
# LENSFUN DISTORTION
# ============================================================================

def apply_lensfun_distortion(
    image: np.ndarray,
    camera,
    lens,
    metadata: ImageMetadata,
    options: LensCorrectionOptions,
) -> np.ndarray:

    height, width = image.shape[:2]

    focal_length = (
        metadata.focal_length
        if metadata.focal_length
        else estimate_focal_length(lens)
    )

    aperture = (
        metadata.aperture
        if metadata.aperture
        else estimate_aperture(lens)
    )

    distance = (
        metadata.focus_distance
        if metadata.focus_distance
        else 10.0
    )

    crop_factor = getattr(
        camera,
        "crop_factor",
        1.0,
    )

    print(
        "[Lensfun]"
        f" focal={focal_length}"
        f" aperture={aperture}"
        f" distance={distance}"
    )

    modifier = lensfunpy.Modifier(
        lens,
        crop_factor,
        width,
        height,
    )

    flags = 0

    # Lensfun flags vary slightly between releases,
    # so only use them if available.

    if options.vignetting:

        flags |= getattr(
            lensfunpy.ModifyFlags,
            "VIGNETTING",
            0,
        )

    if options.tca:

        flags |= getattr(
            lensfunpy.ModifyFlags,
            "TCA",
            0,
        )

    try:

        modifier.initialize(
            focal_length,
            aperture,
            distance,
            pixel_format=np.uint8,
            flags=flags,
        )

    except TypeError:

        # Compatibility with older lensfunpy.
        modifier.initialize(
            focal_length,
            aperture,
            distance,
            pixel_format=np.uint8,
        )

    # ------------------------------------------------------------------------
    # VIGNETTING
    # ------------------------------------------------------------------------

    result = image.copy()

    if options.vignetting:

        print(
            "[Lensfun] Applying vignetting correction..."
        )

        image8 = np.round(
            result * 255.0
        ).astype(np.uint8)

        did_apply = False

        try:

            did_apply = modifier.apply_color_modification(
                image8
            )

        except Exception as exc:

            print(
                "[Lensfun] Vignetting unavailable:",
                exc,
            )

        if did_apply:

            result = (
                image8.astype(np.float32)
                / 255.0
            )

        else:

            print(
                "[Lensfun] No vignetting profile data."
            )

    # ------------------------------------------------------------------------
    # TCA
    # ------------------------------------------------------------------------

    if options.tca:

        print(
            "[Lensfun] Applying TCA correction..."
        )

        try:

            coords = (
                modifier
                .apply_subpixel_distortion()
            )

            image8 = np.round(
                result * 255.0
            ).astype(np.uint8)

            corrected = np.empty_like(
                image8
            )

            # Lensfun returns one map per RGB
            # channel.

            for channel in range(3):

                corrected[..., channel] = (
                    cv2.remap(
                        image8[..., channel],
                        coords[..., channel, :],
                        None,
                        options.interpolation,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                )

            result = (
                corrected.astype(np.float32)
                / 255.0
            )

        except Exception as exc:

            print(
                "[Lensfun] TCA unavailable:",
                exc,
            )

    # ------------------------------------------------------------------------
    # GEOMETRIC DISTORTION
    # ------------------------------------------------------------------------

    if options.distortion:

        print(
            "[Lensfun] Applying geometric distortion correction..."
        )

        try:

            coords = (
                modifier
                .apply_geometry_distortion()
            )

            image8 = np.round(
                result * 255.0
            ).astype(np.uint8)

            corrected = cv2.remap(
                image8,
                coords,
                None,
                options.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
            )

            result = (
                corrected.astype(np.float32)
                / 255.0
            )

        except Exception as exc:

            raise RuntimeError(
                "Lensfun distortion correction failed: "
                f"{exc}"
            ) from exc

    return finite_image(result)


# ============================================================================
# FOCAL/APERTURE FALLBACKS
# ============================================================================

def estimate_focal_length(
    lens,
) -> float:

    value = getattr(
        lens,
        "min_focal",
        None,
    )

    if value is not None:
        return float(value)

    return 50.0


def estimate_aperture(
    lens,
) -> float:

    value = getattr(
        lens,
        "max_aperture",
        None,
    )

    if value is not None:
        return float(value)

    return 4.0


# ============================================================================
# MANUAL OPENCV DISTORTION
# ============================================================================

def apply_manual_distortion(
    image: np.ndarray,
    settings: ManualDistortion,
    interpolation: int,
) -> np.ndarray:

    height, width = image.shape[:2]

    print()
    print(
        "[Manual]"
        f" k1={settings.k1}"
        f" k2={settings.k2}"
        f" k3={settings.k3}"
        f" p1={settings.p1}"
        f" p2={settings.p2}"
    )

    print(
        "[Manual]"
        f" center=({settings.center_x:.4f},"
        f"{settings.center_y:.4f})"
    )

    print(
        "[Manual]"
        f" strength={settings.strength}"
    )

    # ------------------------------------------------------------------------
    # Camera matrix
    # ------------------------------------------------------------------------

    # Focal scale relative to image dimensions.
    #
    # This does not represent physical focal length.
    # It determines how aggressively the distortion model
    # is applied.

    base_focal = max(
        width,
        height,
    ) * settings.focal_scale

    cx = (
        settings.center_x
        * width
    )

    cy = (
        settings.center_y
        * height
    )

    camera_matrix = np.array(
        [
            [
                base_focal,
                0.0,
                cx,
            ],
            [
                0.0,
                base_focal,
                cy,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # Distortion coefficients
    # ------------------------------------------------------------------------

    # OpenCV:
    #
    # k1, k2, p1, p2, k3

    distortion = np.array(
        [
            settings.k1 * settings.strength,
            settings.k2 * settings.strength,
            settings.p1 * settings.strength,
            settings.p2 * settings.strength,
            settings.k3 * settings.strength,
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # Generate mapping
    # ------------------------------------------------------------------------

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )

    corrected = cv2.remap(
        image,
        map1,
        map2,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return finite_image(corrected)


# ============================================================================
# AUTO CROP
# ============================================================================

def calculate_valid_crop(
    original: np.ndarray,
    corrected: np.ndarray,
) -> Tuple[int, int, int, int]:

    # Detect pixels that are not black.
    luminance = (
        0.2126 * corrected[..., 0]
        + 0.7152 * corrected[..., 1]
        + 0.0722 * corrected[..., 2]
    )

    valid = luminance > 0.005

    ys, xs = np.where(valid)

    if len(xs) == 0 or len(ys) == 0:

        return (
            0,
            0,
            corrected.shape[1],
            corrected.shape[0],
        )

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    # Conservative crop.
    #
    # The largest valid rectangle would require
    # additional geometry processing. This approach
    # removes obvious black borders without over-cropping.

    return (
        x0,
        y0,
        x1,
        y1,
    )


def crop_to_valid_area(
    image: np.ndarray,
    alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    x0, y0, x1, y1 = calculate_valid_crop(
        image,
        image,
    )

    cropped = image[
        y0:y1,
        x0:x1,
    ]

    if alpha is not None:

        alpha = alpha[
            y0:y1,
            x0:x1,
        ]

    return cropped, alpha


# ============================================================================
# DEBUG GRID
# ============================================================================

def create_debug_grid(
    image: np.ndarray,
    spacing: int = 100,
) -> np.ndarray:

    result = image.copy()

    height, width = result.shape[:2]

    # Vertical lines
    for x in range(
        0,
        width,
        spacing,
    ):

        cv2.line(
            result,
            (x, 0),
            (x, height - 1),
            (1.0, 0.0, 0.0),
            1,
        )

    # Horizontal lines
    for y in range(
        0,
        height,
        spacing,
    ):

        cv2.line(
            result,
            (0, y),
            (width - 1, y),
            (1.0, 0.0, 0.0),
            1,
        )

    return result


# ============================================================================
# PROFILE LISTING
# ============================================================================

def list_cameras(
    query: str,
):

    db = create_database()

    cameras = db.find_cameras(
        "",
        query,
    )

    print()

    if not cameras:

        print(
            "No cameras found."
        )

        return

    print(
        f"Found {len(cameras)} cameras:"
    )

    for camera in cameras:

        print(
            f"- {camera}"
        )


def list_lenses(
    query: str,
):

    db = create_database()

    # Lensfun's Python API differs slightly
    # across versions. Search through cameras
    # and their compatible lenses.

    found = []

    cameras = db.cameras

    query_lower = query.lower()

    for camera in cameras:

        try:

            lenses = db.find_lenses(
                camera,
                "",
                query,
            )

        except Exception:

            continue

        for lens in lenses:

            if lens not in found:

                found.append(lens)

    print()

    if not found:

        print(
            "No lenses found."
        )

        return

    print(
        f"Found {len(found)} lenses:"
    )

    for lens in found:

        name = str(lens)

        if (
            query_lower
            in name.lower()
        ):

            print(
                f"- {name}"
            )


# ============================================================================
# PROFILE SELECTION
# ============================================================================

def load_profile_from_cli(
    metadata: ImageMetadata,
    camera_maker: Optional[str],
    camera_model: Optional[str],
    lens_maker: Optional[str],
    lens_model: Optional[str],
):

    if camera_maker:
        metadata.camera_make = camera_maker

    if camera_model:
        metadata.camera_model = camera_model

    if lens_maker:
        metadata.lens_make = lens_maker

    if lens_model:
        metadata.lens_model = lens_model

    return find_lensfun_profile(
        metadata
    )


# ============================================================================
# PROCESS
# ============================================================================

def process(
    input_path: str,
    output_path: str,
    options: LensCorrectionOptions,
    manual: ManualDistortion,
    camera_maker: Optional[str] = None,
    camera_model: Optional[str] = None,
    lens_maker: Optional[str] = None,
    lens_model: Optional[str] = None,
):

    print()
    print("=" * 72)
    print("AUTO PHOTO EDITOR - LENS CORRECTION")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    metadata = read_exif(
        input_path
    )

    metadata.camera_make = (
        camera_maker
        or metadata.camera_make
    )

    metadata.camera_model = (
        camera_model
        or metadata.camera_model
    )

    metadata.lens_make = (
        lens_maker
        or metadata.lens_make
    )

    metadata.lens_model = (
        lens_model
        or metadata.lens_model
    )

    print_metadata(
        metadata
    )

    # ------------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------------

    image, alpha = load_image(
        input_path
    )

    original = image.copy()

    # ------------------------------------------------------------------------
    # Debug original
    # ------------------------------------------------------------------------

    if options.debug_grid:

        debug = create_debug_grid(
            original
        )

        debug_path = (
            os.path.splitext(
                output_path
            )[0]
            + "_before_grid.jpg"
        )

        save_image(
            debug_path,
            debug,
        )

        print(
            "[DEBUG] Saved:",
            debug_path,
        )

    # ------------------------------------------------------------------------
    # Lensfun
    # ------------------------------------------------------------------------

    if options.use_lensfun and not options.manual:
        if not metadata.camera_model and not camera_model:
            print(
                "[Lensfun] Camera metadata is unavailable; "
                "skipping automatic correction. Use --manual with distortion "
                "coefficients or provide camera/lens metadata."
            )
        else:
            db, camera, lens = load_profile_from_cli(
                metadata,
                camera_maker,
                camera_model,
                lens_maker,
                lens_model,
            )

            print_lens_profile(camera, lens)

            image = apply_lensfun_distortion(
                image,
                camera,
                lens,
                metadata,
                options,
            )

    # ------------------------------------------------------------------------
    # Manual
    # ------------------------------------------------------------------------

    if options.manual:

        image = apply_manual_distortion(
            image,
            manual,
            options.interpolation,
        )

    # ------------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------------

    if options.auto_crop:

        # We only crop if distortion correction
        # produced black borders.

        before_shape = image.shape

        image, alpha = crop_to_valid_area(
            image,
            alpha,
        )

        print(
            "[Crop]",
            before_shape[:2],
            "->",
            image.shape[:2],
        )

    # ------------------------------------------------------------------------
    # Debug after
    # ------------------------------------------------------------------------

    if options.debug_grid:

        debug = create_debug_grid(
            image
        )

        debug_path = (
            os.path.splitext(
                output_path
            )[0]
            + "_after_grid.jpg"
        )

        save_image(
            debug_path,
            debug,
        )

        print(
            "[DEBUG] Saved:",
            debug_path,
        )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_image(
        output_path,
        image,
        alpha,
    )

    print()
    print(
        "[SUCCESS]"
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "Size:",
        f"{image.shape[1]} x {image.shape[0]}",
    )

    print("=" * 72)
    print()


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic and manual lens "
            "distortion correction."
        )
    )

    # ------------------------------------------------------------------------
    # Normal processing
    # ------------------------------------------------------------------------

    parser.add_argument(
        "input",
        nargs="?",
        help="Input image.",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output image.",
    )

    # ------------------------------------------------------------------------
    # Camera / lens overrides
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--camera-maker",
        default=None,
        help="Override camera maker.",
    )

    parser.add_argument(
        "--camera",
        dest="camera_model",
        default=None,
        help="Override camera model.",
    )

    parser.add_argument(
        "--lens-maker",
        default=None,
        help="Override lens maker.",
    )

    parser.add_argument(
        "--lens",
        dest="lens_model",
        default=None,
        help="Override lens model.",
    )

    # ------------------------------------------------------------------------
    # Lensfun
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--no-lensfun",
        action="store_true",
        help="Disable Lensfun.",
    )

    parser.add_argument(
        "--no-distortion",
        action="store_true",
        help="Disable geometric distortion correction.",
    )

    parser.add_argument(
        "--tca",
        action="store_true",
        help="Correct transverse chromatic aberration.",
    )

    parser.add_argument(
        "--vignetting",
        action="store_true",
        help="Correct lens vignetting.",
    )

    # ------------------------------------------------------------------------
    # Manual
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual OpenCV distortion.",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=0.0,
        help="Radial distortion K1.",
    )

    parser.add_argument(
        "--k2",
        type=float,
        default=0.0,
        help="Radial distortion K2.",
    )

    parser.add_argument(
        "--k3",
        type=float,
        default=0.0,
        help="Radial distortion K3.",
    )

    parser.add_argument(
        "--p1",
        type=float,
        default=0.0,
        help="Tangential distortion P1.",
    )

    parser.add_argument(
        "--p2",
        type=float,
        default=0.0,
        help="Tangential distortion P2.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Manual distortion strength.",
    )

    parser.add_argument(
        "--center-x",
        type=float,
        default=0.5,
        help="Optical center X as fraction of width.",
    )

    parser.add_argument(
        "--center-y",
        type=float,
        default=0.5,
        help="Optical center Y as fraction of height.",
    )

    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Manual virtual focal scaling.",
    )

    # ------------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not automatically crop.",
    )

    # ------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug information.",
    )

    parser.add_argument(
        "--debug-grid",
        action="store_true",
        help="Save before/after grid images.",
    )

    # ------------------------------------------------------------------------
    # Database tools
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--search-lens",
        default=None,
        help="Search Lensfun database.",
    )

    parser.add_argument(
        "--list-cameras",
        default=None,
        help="List matching cameras.",
    )

    parser.add_argument(
        "--list-lenses",
        default=None,
        help="List matching lenses.",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Database utilities
    # ------------------------------------------------------------------------

    if args.search_lens:

        list_lenses(
            args.search_lens
        )

        return 0

    if args.list_cameras:

        list_cameras(
            args.list_cameras
        )

        return 0

    if args.list_lenses:

        list_lenses(
            args.list_lenses
        )

        return 0

    if not args.input or not args.output:

        parser.error(
            "input and output are required "
            "unless using --search-lens, "
            "--list-cameras or --list-lenses."
        )

    # ------------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------------

    options = LensCorrectionOptions(
        use_lensfun=not args.no_lensfun,

        distortion=not args.no_distortion,

        tca=args.tca,

        vignetting=args.vignetting,

        manual=args.manual,

        auto_crop=not args.no_crop,

        debug=args.debug,

        debug_grid=args.debug_grid,
    )

    # ------------------------------------------------------------------------
    # Manual settings
    # ------------------------------------------------------------------------

    manual = ManualDistortion(

        k1=args.k1,
        k2=args.k2,
        k3=args.k3,

        p1=args.p1,
        p2=args.p2,

        center_x=args.center_x,
        center_y=args.center_y,

        focal_scale=args.focal_scale,

        strength=args.strength,
    )

    try:

        process(
            input_path=args.input,
            output_path=args.output,
            options=options,
            manual=manual,

            camera_maker=args.camera_maker,
            camera_model=args.camera_model,

            lens_maker=args.lens_maker,
            lens_model=args.lens_model,
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\nCancelled."
        )

        return 130

    except Exception as exc:

        print(
            "\nERROR:",
            exc,
            file=sys.stderr,
        )

        if args.debug:

            import traceback

            traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )