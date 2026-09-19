from __future__ import annotations

from .core import *
from .profiles import *
from .correction import *

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
