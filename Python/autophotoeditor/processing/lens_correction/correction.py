from __future__ import annotations

from .core import *
from .profiles import *

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


