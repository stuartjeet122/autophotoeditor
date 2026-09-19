from __future__ import annotations

from .core import *
from .parsing import *
from .color import *
from .io import *
from .adjustments import *

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Apply Lightroom XMP-inspired "
            "adjustments to an image."
        )
    )

    parser.add_argument(
        "xmp",
        nargs="?",
        help="Path to XMP preset.",
    )

    parser.add_argument(
        "input",
        help="Input image.",
    )

    parser.add_argument(
        "output",
        help="Output image.",
    )

    # ------------------------------------------------------------------------
    # Category selectors
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--exposure-from-xmp",
        action="store_true",
        help="Use XMP exposure only.",
    )

    parser.add_argument(
        "--white-balance",
        action="store_true",
        help="Apply XMP white balance.",
    )

    parser.add_argument(
        "--tone",
        action="store_true",
        help="Apply tone settings.",
    )

    parser.add_argument(
        "--point-curve",
        action="store_true",
        help="Apply XMP point curve.",
    )

    parser.add_argument(
        "--hsl",
        action="store_true",
        help="Apply HSL settings.",
    )

    parser.add_argument(
        "--color",
        action="store_true",
        help="Apply global color settings.",
    )

    parser.add_argument(
        "--effects",
        action="store_true",
        help="Apply effects.",
    )

    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply noise reduction.",
    )

    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply sharpening.",
    )

    # ------------------------------------------------------------------------
    # Tone parameters
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Override exposure.",
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="Override contrast.",
    )

    parser.add_argument(
        "--highlights",
        type=float,
        default=None,
        help="Override highlights.",
    )

    parser.add_argument(
        "--shadows",
        type=float,
        default=None,
        help="Override shadows.",
    )

    parser.add_argument(
        "--whites",
        type=float,
        default=None,
        help="Override whites.",
    )

    parser.add_argument(
        "--blacks",
        type=float,
        default=None,
        help="Override blacks.",
    )

    # ------------------------------------------------------------------------
    # Color parameters
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--vibrance",
        type=float,
        default=None,
        help="Override vibrance.",
    )

    parser.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="Override global saturation.",
    )

    # ------------------------------------------------------------------------
    # WB parameters
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature.",
    )

    parser.add_argument(
        "--tint",
        type=float,
        default=None,
        help="Override tint.",
    )

    # ------------------------------------------------------------------------
    # Effects parameters
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--clarity",
        type=float,
        default=None,
        help="Override clarity.",
    )

    parser.add_argument(
        "--texture",
        type=float,
        default=None,
        help="Override texture.",
    )

    parser.add_argument(
        "--dehaze",
        type=float,
        default=None,
        help="Override dehaze.",
    )

    # ------------------------------------------------------------------------
    # HSL parameters
    # ------------------------------------------------------------------------

    for channel in HSL_CHANNELS:

        parser.add_argument(
            f"--hsl-{channel}",
            type=str,
            default=None,
            metavar="H,S,L",
            help=(
                f"{channel.capitalize()} "
                "HSL override."
            ),
        )

    # ------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate stages.",
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Debug directory.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=97,
        help="JPEG output quality.",
    )

    return parser


# ============================================================================
# PROCESSING ARGUMENT DETECTION
# ============================================================================

def any_processing_argument_requested(
    args: argparse.Namespace,
) -> bool:

    # Explicit category selectors.
    if any([
        args.exposure_from_xmp,
        args.white_balance,
        args.tone,
        args.point_curve,
        args.hsl,
        args.color,
        args.effects,
        args.noise,
        args.sharpen,
    ]):
        return True

    # Individual parameters.
    if any([
        args.exposure is not None,
        args.contrast is not None,
        args.highlights is not None,
        args.shadows is not None,
        args.whites is not None,
        args.blacks is not None,
        args.vibrance is not None,
        args.saturation is not None,
        args.temperature is not None,
        args.tint is not None,
        args.clarity is not None,
        args.texture is not None,
        args.dehaze is not None,
    ]):
        return True

    # Individual HSL channels.
    for channel in HSL_CHANNELS:

        if getattr(
            args,
            f"hsl_{channel}",
        ) is not None:
            return True

    return False


def build_pipeline_options(
    args: argparse.Namespace,
) -> PipelineOptions:

    processing_argument_given = (
        any_processing_argument_requested(
            args
        )
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # If NO processing argument is supplied,
    # apply every supported category.
    # ------------------------------------------------------------------------

    if not processing_argument_given:

        return PipelineOptions(
            use_exposure=True,
            use_white_balance=True,
            use_tone=True,
            use_point_curve=True,
            use_hsl=True,
            use_color=True,
            use_effects=True,
            use_noise=True,
            use_sharpening=True,
            debug=args.debug,
        )

    # ------------------------------------------------------------------------
    # Otherwise only explicitly requested categories.
    #
    # Individual parameters automatically enable their category.
    # ------------------------------------------------------------------------

    return PipelineOptions(
        use_exposure=(
            args.exposure is not None
            or args.exposure_from_xmp
        ),

        use_white_balance=(
            args.white_balance
            or args.temperature is not None
            or args.tint is not None
        ),

        use_tone=(
            args.tone
            or any([
                args.contrast is not None,
                args.highlights is not None,
                args.shadows is not None,
                args.whites is not None,
                args.blacks is not None,
            ])
        ),

        use_point_curve=args.point_curve,

        use_hsl=(
            args.hsl
            or any(
                getattr(
                    args,
                    f"hsl_{channel}",
                ) is not None
                for channel in HSL_CHANNELS
            )
        ),

        use_color=(
            args.color
            or args.vibrance is not None
            or args.saturation is not None
        ),

        use_effects=(
            args.effects
            or args.clarity is not None
            or args.texture is not None
            or args.dehaze is not None
        ),

        use_noise=args.noise,

        use_sharpening=args.sharpen,

        debug=args.debug,
    )


# ============================================================================
# CLI OVERRIDES
# ============================================================================

def apply_cli_overrides(
    settings: XMPSettings,
    args: argparse.Namespace,
) -> XMPSettings:

    # ------------------------------------------------------------------------
    # Tone
    # ------------------------------------------------------------------------

    if args.exposure is not None:
        settings.tone.exposure = args.exposure

    if args.contrast is not None:
        settings.tone.contrast = args.contrast

    if args.highlights is not None:
        settings.tone.highlights = args.highlights

    if args.shadows is not None:
        settings.tone.shadows = args.shadows

    if args.whites is not None:
        settings.tone.whites = args.whites

    if args.blacks is not None:
        settings.tone.blacks = args.blacks

    # ------------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------------

    if args.vibrance is not None:
        settings.color.vibrance = args.vibrance

    if args.saturation is not None:
        settings.color.saturation = args.saturation

    # ------------------------------------------------------------------------
    # WB
    # ------------------------------------------------------------------------

    if args.temperature is not None:
        settings.wb.temperature = args.temperature

    if args.tint is not None:
        settings.wb.tint = args.tint

    # ------------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------------

    if args.clarity is not None:
        settings.effects.clarity = args.clarity

    if args.texture is not None:
        settings.effects.texture = args.texture

    if args.dehaze is not None:
        settings.effects.dehaze = args.dehaze

    # ------------------------------------------------------------------------
    # HSL
    # ------------------------------------------------------------------------

    for channel in HSL_CHANNELS:

        value = getattr(
            args,
            f"hsl_{channel}",
        )

        if value is not None:

            settings.hsl[channel] = (
                parse_hsl_override(value)
            )

    return settings


# ============================================================================
# SETTINGS REPORT
# ============================================================================

def print_settings(
    settings: XMPSettings,
    options: PipelineOptions,
):

    print()
    print("=" * 72)
    print("XMP SETTINGS")
    print("=" * 72)

    print(
        f"Exposure:   {settings.tone.exposure:+.2f}"
    )

    print(
        f"Contrast:   {settings.tone.contrast:+.1f}"
    )

    print(
        f"Highlights: {settings.tone.highlights:+.1f}"
    )

    print(
        f"Shadows:    {settings.tone.shadows:+.1f}"
    )

    print(
        f"Whites:     {settings.tone.whites:+.1f}"
    )

    print(
        f"Blacks:     {settings.tone.blacks:+.1f}"
    )

    print(
        f"Temperature:{settings.wb.temperature:+.1f}"
    )

    print(
        f"Tint:       {settings.wb.tint:+.1f}"
    )

    print(
        f"Vibrance:   {settings.color.vibrance:+.1f}"
    )

    print(
        f"Saturation: {settings.color.saturation:+.1f}"
    )

    print(
        f"Clarity:    {settings.effects.clarity:+.1f}"
    )

    print(
        f"Texture:    {settings.effects.texture:+.1f}"
    )

    print(
        f"Dehaze:     {settings.effects.dehaze:+.1f}"
    )

    print()
    print("HSL:")

    for channel in HSL_CHANNELS:

        value = settings.hsl[channel]

        print(
            f"  {channel:<8}"
            f"H={value.hue:+.1f} "
            f"S={value.saturation:+.1f} "
            f"L={value.luminance:+.1f}"
        )

    print()
    print(
        "Point curve:",
        len(settings.point_curve.points),
        "points",
    )

    print(
        "Camera profile:",
        settings.camera_profile,
    )

    print(
        "Process version:",
        settings.process_version,
    )

    print()
    print("PIPELINE:")

    print(
        "  Exposure:      ",
        options.use_exposure,
    )

    print(
        "  White balance: ",
        options.use_white_balance,
    )

    print(
        "  Tone:          ",
        options.use_tone,
    )

    print(
        "  Point curve:   ",
        options.use_point_curve,
    )

    print(
        "  HSL:           ",
        options.use_hsl,
    )

    print(
        "  Color:         ",
        options.use_color,
    )

    print(
        "  Effects:       ",
        options.use_effects,
    )

    print(
        "  Noise:         ",
        options.use_noise,
    )

    print(
        "  Sharpen:       ",
        options.use_sharpening,
    )

    if settings.unsupported:

        print()
        print("UNSUPPORTED / APPROXIMATED FEATURES:")

        for feature in settings.unsupported:
            print(
                f"  - {feature}"
            )

    print("=" * 72)
    print()


