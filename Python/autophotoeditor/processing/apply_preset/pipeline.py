from __future__ import annotations

from .core import *
from .parsing import *
from .color import *
from .io import *
from .adjustments import *
from .cli import *

def process_image(
    image: np.ndarray,
    alpha: Optional[np.ndarray],
    settings: XMPSettings,
    options: PipelineOptions,
    debug_dir: Optional[str] = None,
) -> np.ndarray:

    result = image.astype(
        np.float32,
        copy=True,
    )

    stage_index = 0

    def stage(
        name: str,
        current: np.ndarray,
    ) -> np.ndarray:

        nonlocal stage_index

        stage_index += 1

        current = finite_image(
            current
        )

        print(
            f"[{stage_index:02d}] "
            f"{name:<24} "
            f"{image_statistics(current)}"
        )

        if options.debug:

            directory = (
                debug_dir
                or "debug_stages"
            )

            save_debug_stage(
                current,
                alpha,
                directory,
                f"{stage_index:02d}_{name}",
            )

        return current

    # ------------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------------

    if options.use_exposure:

        result = apply_exposure(
            result,
            settings.tone.exposure,
        )

        result = stage(
            "exposure",
            result,
        )

    # ------------------------------------------------------------------------
    # White balance
    # ------------------------------------------------------------------------

    if options.use_white_balance:

        result = apply_white_balance(
            result,
            settings.wb.temperature,
            settings.wb.tint,
        )

        result = stage(
            "white_balance",
            result,
        )

    # ------------------------------------------------------------------------
    # Tone
    # ------------------------------------------------------------------------

    if options.use_tone:

        result = apply_tone(
            result,
            settings.tone,
        )

        result = stage(
            "tone",
            result,
        )

    # ------------------------------------------------------------------------
    # Point curve
    # ------------------------------------------------------------------------

    if options.use_point_curve:

        result = apply_point_curve(
            result,
            settings.point_curve,
        )

        result = stage(
            "point_curve",
            result,
        )

    # ------------------------------------------------------------------------
    # HSL
    # ------------------------------------------------------------------------

    if options.use_hsl:

        result = apply_hsl(
            result,
            settings.hsl,
        )

        result = stage(
            "hsl",
            result,
        )

    # ------------------------------------------------------------------------
    # Global color
    # ------------------------------------------------------------------------

    if options.use_color:

        result = apply_color(
            result,
            settings.color,
        )

        result = stage(
            "color",
            result,
        )

    # ------------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------------

    if options.use_effects:

        result = apply_effects(
            result,
            settings.effects,
        )

        result = stage(
            "effects",
            result,
        )

    # ------------------------------------------------------------------------
    # Noise reduction
    # ------------------------------------------------------------------------

    if options.use_noise:

        result = apply_noise_reduction(
            result,
            settings.noise,
        )

        result = stage(
            "noise_reduction",
            result,
        )

    # ------------------------------------------------------------------------
    # Sharpen
    # ------------------------------------------------------------------------

    if options.use_sharpening:

        result = apply_sharpening(
            result,
            settings.sharpen,
        )

        result = stage(
            "sharpen",
            result,
        )

    return finite_image(result)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    try:

        print()
        print("=" * 72)
        print("AUTO PHOTO EDITOR")
        print("=" * 72)

        print(
            "Input :",
            args.input,
        )

        print(
            "Output:",
            args.output,
        )

        if args.xmp:
            print(
                "XMP   :",
                args.xmp,
            )
        else:
            print(
                "XMP   : none",
            )

        # --------------------------------------------------------------------
        # Determine processing behavior.
        # --------------------------------------------------------------------

        options = build_pipeline_options(
            args
        )

        processing_argument_given = (
            any_processing_argument_requested(
                args
            )
        )

        if not processing_argument_given:

            print(
                "Mode  : ALL supported XMP settings"
            )

        else:

            print(
                "Mode  : SELECTED processing categories"
            )

        # --------------------------------------------------------------------
        # Load XMP.
        # --------------------------------------------------------------------

        if args.xmp:

            settings = parse_xmp(
                args.xmp
            )

        else:

            settings = XMPSettings()

        # --------------------------------------------------------------------
        # Apply CLI overrides.
        # --------------------------------------------------------------------

        settings = apply_cli_overrides(
            settings,
            args,
        )

        # --------------------------------------------------------------------
        # Print settings.
        # --------------------------------------------------------------------

        print_settings(
            settings,
            options,
        )

        # --------------------------------------------------------------------
        # Load image.
        # --------------------------------------------------------------------

        print(
            "[INFO] Loading image..."
        )

        image, alpha = load_image(
            args.input
        )

        print(
            "[INFO] Image:",
            image.shape,
            image.dtype,
        )

        if alpha is not None:
            print(
                "[INFO] Alpha channel preserved."
            )

        # --------------------------------------------------------------------
        # Process.
        # --------------------------------------------------------------------

        result = process_image(
            image,
            alpha,
            settings,
            options,
            debug_dir=args.debug_dir,
        )

        # --------------------------------------------------------------------
        # Save.
        # --------------------------------------------------------------------

        print(
            "[INFO] Saving..."
        )

        save_image(
            args.output,
            result,
            alpha,
            jpeg_quality=args.jpeg_quality,
        )

        print()
        print(
            f"[SUCCESS] Saved: {args.output}"
        )
        print()

        return 0

    except KeyboardInterrupt:

        print(
            "\n[ERROR] Cancelled."
        )

        return 130

    except Exception as exc:

        print(
            f"\n[ERROR] {exc}",
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
