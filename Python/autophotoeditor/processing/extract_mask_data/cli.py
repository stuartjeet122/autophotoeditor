from __future__ import annotations

from .shared import *
from .pipeline import *

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Color-science / OpenCV automatic "
            "per-mask Lightroom-style enhancement analyzer."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Original input image.",
    )

    parser.add_argument(
        "masks",
        type=Path,
        help="Mask extractor JSON.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output enhancement JSON.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Overall correction strength. "
            "1.0 = normal automatic correction."
        ),
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    return parser


# ============================================================
# VALIDATION
# ============================================================

def validate_args(
    args: argparse.Namespace,
) -> None:

    if not (
        0.0 <= args.strength <= 2.0
    ):
        fail(
            "--strength must be between 0 and 2."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    configure_logging(
        args.verbose
    )

    validate_args(
        args
    )

    try:

        run(
            input_path=args.input,
            masks_json=args.masks,
            output_json=args.output,
            strength=args.strength,
            pretty=args.pretty,
        )

    except KeyboardInterrupt:

        logger.error(
            "Interrupted."
        )

        raise SystemExit(130)

    except Exception as exc:

        logger.exception(
            "Fatal error: %s",
            exc,
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
