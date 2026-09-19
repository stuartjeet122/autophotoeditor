from __future__ import annotations

from .core import *
from .pipeline import *

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive semantic-mask photo "
            "enhancement with neutral illumination "
            "correction."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Original input image",
    )

    parser.add_argument(
        "masks_json",
        type=Path,
        help="Mask JSON",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Enhanced output image",
    )

    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing PNG masks."
        ),
    )

    parser.add_argument(
        "--person-masks-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing "
            "person_001/soft, person_002/soft, etc."
        ),
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Global enhancement strength "
            "(0.0-1.0). Default: 1.0"
        ),
    )

    parser.add_argument(
        "--feather",
        type=float,
        default=2.0,
        help=(
            "Mask feather radius in pixels. "
            "Default: 2.0"
        ),
    )

    parser.add_argument(
        "--save-mask-results",
        type=Path,
        default=None,
        help=(
            "Directory for individual mask "
            "previews."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    try:

        strength = float(
            np.clip(
                args.strength,
                0.0,
                1.0,
            )
        )

        feather = max(
            0.0,
            float(
                args.feather
            ),
        )

        process(
            image_path=args.input,
            json_path=args.masks_json,
            output_path=args.output,
            mask_dir=args.mask_dir,
            global_strength=strength,
            feather_radius=feather,
            save_mask_results=(
                args.save_mask_results
            ),
            person_masks_dir=(
                args.person_masks_dir
            ),
        )

        return 0

    except KeyboardInterrupt:

        logger.error(
            "Interrupted by user."
        )

        return 130

    except Exception as exc:

        logger.exception(
            "ERROR: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
