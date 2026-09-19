from __future__ import annotations

from .core import *

def imread_unicode(
    path: Path,
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:

    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            image = decode_color(path)

            if flags == cv2.IMREAD_GRAYSCALE:
                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

            return image

        except Exception as exc:
            logger.warning(
                "Could not decode RAW image %s: %s",
                path,
                exc,
            )
            return None

    try:
        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        if data.size == 0:
            return None

        return cv2.imdecode(
            data,
            flags,
        )

    except Exception as exc:
        logger.warning(
            "Could not read image %s: %s",
            path,
            exc,
        )
        return None


def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    quality: int = 95,
) -> bool:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(quality),
        ]

    elif suffix == ".png":
        ext = ".png"
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    elif suffix == ".webp":
        ext = ".webp"
        params = [
            cv2.IMWRITE_WEBP_QUALITY,
            int(quality),
        ]

    elif suffix in {".tif", ".tiff"}:
        ext = suffix
        params = [
            cv2.IMWRITE_TIFF_COMPRESSION,
            1,
        ]

    else:
        logger.error(
            "Unsupported output extension: %s",
            suffix,
        )
        return False

    try:
        ok, encoded = cv2.imencode(
            ext,
            image,
            params,
        )

        if not ok:
            return False

        encoded.tofile(str(path))
        return True

    except Exception as exc:
        logger.error(
            "Could not save %s: %s",
            path,
            exc,
        )
        return False


# ============================================================================
# PATH / JSON HELPERS
# ============================================================================

MASK_FILE_KEYS = {
    "file",
    "path",
    "mask_file",
    "mask_path",
    "png",
    "filename",
    "mask_filename",
    "soft_file",
    "binary_file",
}

MASK_BINARY_KEYS = {
    "binary_png_base64",
    "binary_base64",
    "png_base64",
    "mask_png_base64",
}


def resolve_path(
    raw_path: str,
    json_path: Path,
    mask_dir: Optional[Path],
) -> Optional[Path]:

    if not raw_path:
        return None

    raw = Path(str(raw_path))
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    else:

        if mask_dir is not None:
            candidates.append(
                mask_dir / raw
            )

        candidates.append(
            json_path.parent / raw
        )

        if mask_dir is not None:
            candidates.append(
                mask_dir / raw.name
            )

        candidates.append(
            json_path.parent / raw.name
        )

    for candidate in candidates:

        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            pass

    return None


def extract_mask_entries(
    data: Any,
    json_path: Path,
    mask_dir: Optional[Path],
) -> Dict[str, Path]:

    result: Dict[str, Path] = {}
    embedded_dir = Path(tempfile.mkdtemp(prefix="autophotoeditor_masks_"))

    def add_entry(
        name: str,
        value: Any,
    ) -> None:

        name = str(name).strip().lower()

        if not name:
            return

        if name in {
            "image",
            "input",
            "source",
            "original",
            "photo",
        }:
            return

        path: Optional[Path] = None

        if isinstance(value, str):
            path = resolve_path(
                value,
                json_path,
                mask_dir,
            )

        elif isinstance(value, dict):

            for key in MASK_BINARY_KEYS:

                encoded = value.get(key)

                if not isinstance(encoded, str):
                    continue

                try:
                    binary = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    continue

                path = embedded_dir / f"{name}.png"
                path.write_bytes(binary)
                break

            if path is None:

                for key in MASK_FILE_KEYS:

                    raw = value.get(key)

                    if not isinstance(raw, str):
                        continue

                    path = resolve_path(
                        raw,
                        json_path,
                        mask_dir,
                    )

                    if path is not None:
                        break

        if path is None:
            return

        if path.suffix.lower() != ".png":
            return

        result[name] = path

    if not isinstance(data, dict):
        return result

    masks = data.get("masks")

    if isinstance(masks, dict):

        for name, value in masks.items():
            add_entry(
                name,
                value,
            )

    elif isinstance(masks, list):

        for item in masks:

            if not isinstance(item, dict):
                continue

            name = (
                item.get("name")
                or item.get("label")
                or item.get("class")
                or item.get("mask")
            )

            if name:
                add_entry(
                    str(name),
                    item,
                )

    mask_files = data.get("mask_files")

    if isinstance(mask_files, dict):

        for name, value in mask_files.items():
            add_entry(
                name,
                value,
            )

    elif isinstance(mask_files, list):

        for item in mask_files:

            if isinstance(item, dict):

                name = (
                    item.get("name")
                    or item.get("label")
                    or item.get("class")
                )

                if name:
                    add_entry(
                        str(name),
                        item,
                    )

            elif isinstance(item, str):

                path = resolve_path(
                    item,
                    json_path,
                    mask_dir,
                )

                if (
                    path is not None
                    and path.suffix.lower() == ".png"
                ):
                    result.setdefault(
                        path.stem.lower(),
                        path,
                    )

    return result


def discover_masks_from_directory(
    mask_dir: Path,
) -> Dict[str, Path]:

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}"
        )

    result: Dict[str, Path] = {}

    for path in mask_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() != ".png":
            continue

        name = path.stem.lower()

        if name.startswith(
            (
                "preview",
                "visual",
                "debug",
            )
        ):
            continue

        result[name] = path.resolve()

    return result


# ============================================================================
# MASK LOADING
# ============================================================================

def load_mask(
    path: Path,
    width: int,
    height: int,
) -> Optional[np.ndarray]:

    mask = imread_unicode(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        logger.warning(
            "Could not load mask: %s",
            path,
        )
        return None

    if mask.ndim != 2:
        return None

    mh, mw = mask.shape

    if mw != width or mh != height:
        logger.info(
            "Resizing mask: %s (mask=%dx%d image=%dx%d)",
            path,
            mw,
            mh,
            width,
            height,
        )
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    result = (
        mask.astype(np.float32)
        / 255.0
    )

    result[result < 0.01] = 0.0

    return result


def clean_mask(
    mask: np.ndarray,
) -> np.ndarray:

    mask_u8 = np.clip(
        mask * 255.0,
        0,
        255,
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return (
        mask_u8.astype(np.float32)
        / 255.0
    )


def feather_mask(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:

    if radius <= 0:
        return mask.astype(
            np.float32,
            copy=True,
        )

    sigma = max(
        radius / 2.0,
        0.5,
    )

    result = cv2.GaussianBlur(
        mask.astype(np.float32),
        (0, 0),
        sigma,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


