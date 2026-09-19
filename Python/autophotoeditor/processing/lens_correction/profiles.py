from __future__ import annotations

from .core import *

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


