"""Shared image and JSON file operations used by the HTTP layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import HTTPException


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".nrw", ".orf", ".raf", ".rw2", ".pef", ".srw"}


def is_raw(path: Path | str) -> bool:
    path = Path(path)
    return path.suffix.lower() in RAW_EXTENSIONS


def decode_color(path: Path | str):
    """Decode a standard image or develop a RAW file into 8-bit BGR."""
    path = Path(path)
    if is_raw(path):
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                "RAW support requires the 'rawpy' package. Install requirements-api.txt."
            ) from exc
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=False,
                    output_color=rawpy.ColorSpace.sRGB,
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise RuntimeError(f"Could not decode RAW image: {path}: {exc}") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def read_color(path: Path):
    try:
        return decode_color(path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def result(output: Path, **extra: Any) -> dict[str, Any]:
    return {"output": str(output), **extra}


def preserve_metadata(source: Path, output: Path) -> None:
    """Copy common EXIF, ICC, XMP, and resolution metadata when supported."""
    if is_raw(source) or not output.exists():
        return
    try:
        from PIL import Image

        with Image.open(source) as source_image, Image.open(output) as output_image:
            metadata = {
                key: source_image.info[key]
                for key in ("exif", "icc_profile", "xmp", "dpi")
                if key in source_image.info
            }
            exif = source_image.getexif()
            if exif:
                metadata["exif"] = exif.tobytes()
            if metadata:
                output_image.save(output, **metadata)
    except (OSError, ValueError):
        return


def ensure_source(source: str) -> Path:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Source image not found: {path}")
    return path


def ensure_destination(destination: str) -> Path:
    path = Path(destination).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise HTTPException(status_code=400, detail="Destination must include a file name")
    if is_raw(path):
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_path(
    source: Path,
    action: str,
    temporary: bool = False,
    job: Path | None = None,
    extension: str | None = None,
) -> Path:
    """Build the automatic output name without changing the source format."""
    directory = job if temporary else source.parent
    if temporary and job is None:
        raise ValueError("A job directory is required for temporary output")
    output_extension = extension or source.suffix
    if extension is None and is_raw(source):
        output_extension = ".png"
    return directory / f"{source.stem}_{action}{output_extension}"
