from __future__ import annotations

from .core import *
from .io import *
from .detection import *
from .transforms import *
from .canvas import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic Lightroom-style "
            "photo geometry correction."
        )
    )

    parser.add_argument(
        "input",
        help="Input image",
    )

    parser.add_argument(
        "output",
        help="Output image",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "off",
            "auto",
            "level",
            "vertical",
            "horizontal",
            "full",
        ],
        default="auto",
    )

    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help=(
            "Additional manual rotation in degrees."
        ),
    )

    parser.add_argument(
        "--vertical-strength",
        type=float,
        default=None,
        help=(
            "Vertical perspective strength "
            "-100..100."
        ),
    )

    parser.add_argument(
        "--horizontal-strength",
        type=float,
        default=None,
        help=(
            "Horizontal perspective strength "
            "-100..100."
        ),
    )

    parser.add_argument(
        "--aspect",
        type=float,
        default=0.0,
        help="Horizontal aspect -100..100.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale percentage.",
    )

    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Horizontal offset percentage.",
    )

    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="Vertical offset percentage.",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Maximum structural lines to retain.",
    )

    parser.add_argument(
        "--analysis-size",
        type=int,
        default=DEFAULT_ANALYSIS_SIZE,
        help="Maximum analysis dimension.",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop projective borders.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality.",
    )

    parser.add_argument(
        "--debug-lines",
        metavar="PATH",
        default=None,
        help="Write detected-line debug image.",
    )

    parser.add_argument(
        "--print-transform",
        action="store_true",
        help="Print final 3x3 homography.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    try:
        print(
            "Loading image...",
            flush=True,
        )

        image = read_image(
            args.input
        )

        h, w = image.shape[:2]

        print(
            f"Image: {w}x{h}",
            flush=True,
        )

        # ----------------------------------------------------
        # OFF
        # ----------------------------------------------------

        if args.mode == "off":
            write_image(
                args.output,
                image,
                args.quality,
            )

            print(
                f"Saved: {args.output}"
            )

            return 0

        # ----------------------------------------------------
        # ANALYSE + BUILD
        # ----------------------------------------------------

        # The analysis-size CLI argument is intentionally not
        # passed directly to build_transform because its API is
        # kept compatible with the rest of the application.
        H, info = build_transform(
            image,
            mode=args.mode,
            rotate=args.rotate,
            vertical_strength=args.vertical_strength,
            horizontal_strength=args.horizontal_strength,
            aspect=args.aspect,
            scale=args.scale,
            x=args.x,
            y=args.y,
            max_lines=max(
                20,
                args.max_lines,
            ),
            verbose=True,
        )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        if args.debug_lines:
            print(
                "[debug] Writing geometry debug image...",
                flush=True,
            )

            draw_debug(
                image,
                info,
                args.debug_lines,
            )

        # ----------------------------------------------------
        # MATRIX
        # ----------------------------------------------------

        if args.print_transform:
            np.set_printoptions(
                precision=8,
                suppress=True,
            )

            print(
                "Transform:"
            )

            print(H)

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        print(
            "[4/4] Rendering final image...",
            flush=True,
        )

        Hfit, out_w, out_h = fit_to_canvas(
            H,
            w,
            h,
        )

        if (
            out_w < MIN_OUTPUT_SIZE
            or out_h < MIN_OUTPUT_SIZE
        ):
            raise RuntimeError(
                "Geometry produced an invalid "
                "output size."
            )

        # ----------------------------------------------------
        # RENDER
        # ----------------------------------------------------

        result = cv2.warpPerspective(
            image,
            Hfit,
            (
                out_w,
                out_h,
            ),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
            ),
        )

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        if not args.no_crop:
            valid_mask = render_validity_mask(
                w,
                h,
                Hfit,
                out_w,
                out_h,
            )

            result = crop_using_mask(
                result,
                valid_mask,
            )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        if (
            result is None
            or result.size == 0
            or result.ndim != 3
        ):
            raise RuntimeError(
                "Geometry correction produced "
                "an invalid image."
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        write_image(
            args.output,
            result,
            args.quality,
        )

        print(
            f"Saved: {args.output}"
        )

        print(
            f"Output: "
            f"{result.shape[1]}x"
            f"{result.shape[0]}"
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nCancelled."
        )

        return 130

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
