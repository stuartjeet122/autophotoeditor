"""FastAPI service for the AutoPhotoEditor processing scripts."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse

from autophotoeditor.core.config import AI_MODEL_CACHE, MODEL_PATH
from autophotoeditor.core.image_io import (
    ensure_source,
    output_path,
    preserve_metadata,
    read_color,
    result,
    save_json,
)


# ============================================================
# DESKTOP CONNECTION
# ============================================================

@app.websocket("/ws")
async def desktop_connection(websocket: WebSocket) -> None:
    """Keep an externally managed desktop client connected and informed."""
    await websocket.accept()

    try:
        await websocket.send_json(
            {
                "type": "hello",
                "service": "autophotoeditor",
                "status": "ok",
            }
        )

        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": message.get("timestamp"),
                    }
                )
            elif message_type == "hello":
                await websocket.send_json(
                    {
                        "type": "hello",
                        "service": "autophotoeditor",
                        "status": "ok",
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Unsupported WebSocket message type.",
                    }
                )
    except WebSocketDisconnect:
        return
from autophotoeditor.models.pipeline import PipelineStage
from autophotoeditor.processing import (
    ai_denoise,
    apply_preset,
    auto_enhance,
    extract_image_data,
    geometry_correction,
    lens_correction,
    manual_adjust,
    masked_adaptive_enhance,
)
from autophotoeditor.services.job_events import (
    EVENTS,
    CURRENT_JOB,
    cancel_job,
    check_cancelled,
    configure_logging,
    cancellation_token,
    job_operation,
    publish_progress,
    validate_job_id,
)


# ============================================================
# APPLICATION
# ============================================================

APP_DIR = Path(__file__).resolve().parents[2]

configure_logging()

app = FastAPI(
    title="AutoPhotoEditor API",
    version="1.0.0",
    description=(
        "Image enhancement and correction routes backed by "
        "the local AutoPhotoEditor processing scripts."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_error_response(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": str(exc.detail),
            },
        },
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_response(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_response(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "message": str(exc),
            },
        },
    )


# ============================================================
# PATH / OUTPUT HELPERS
# ============================================================

def _source(source: str) -> Path:
    return ensure_source(source)


def _job_dir(job_id: Optional[str] = None) -> Path:
    path = (
        Path(tempfile.gettempdir())
        / "autophotoeditor"
        / (job_id or uuid.uuid4().hex)
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _output(
    source: Path,
    action: str,
    temporary: bool = False,
    job: Optional[Path] = None,
    extension: Optional[str] = None,
) -> Path:
    return output_path(
        source,
        action,
        temporary=temporary,
        job=job,
        extension=extension,
    )


def _requested_output(
    source: Path,
    action: str,
    temporary: bool,
    job: Optional[Path],
    destination: Optional[str],
    extension: Optional[str] = None,
) -> Path:
    """
    API results are returned in the response.

    Never write API results beside the input image.
    """
    return _output(
        source,
        action,
        temporary=True,
        job=job or _job_dir(CURRENT_JOB.get()),
        extension=extension,
    )


def _destination_output(
    source: Path,
    destination: str,
    action: str,
) -> Path:
    """
    Create a private temporary output.

    API destinations are never persisted directly by the API layer.
    """
    directory = _job_dir(CURRENT_JOB.get())

    raw_extensions = {
        ".arw",
        ".cr2",
        ".cr3",
        ".dng",
        ".nef",
        ".nrw",
        ".orf",
        ".raf",
        ".rw2",
        ".pef",
        ".srw",
    }

    extension = (
        ".png"
        if source.suffix.lower() in raw_extensions
        else source.suffix
    )

    return directory / f"{source.stem}_{action}{extension}"


def _read_color(path: Path) -> np.ndarray:
    return read_color(path)


# ============================================================
# BINARY IMAGE HELPERS
# ============================================================

def _binary_source(
    image: bytes,
    job_id: Optional[str],
) -> tuple[Path, Path]:
    job = _job_dir(job_id)

    source = job / "input.png"

    source.write_bytes(image)

    if cv2.imread(
        str(source),
        cv2.IMREAD_COLOR,
    ) is None:
        raise HTTPException(
            status_code=400,
            detail="Request body is not a valid image",
        )

    return source, job


def _binary_path(
    image: bytes,
    job_id: Optional[str],
) -> tuple[Path, Path]:
    return _binary_source(
        image,
        job_id,
    )


def _image_artifact(
    output: Path,
) -> dict[str, str]:
    return {
        "kind": "image",
        "encoding": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(
            output.read_bytes()
        ).decode("ascii"),
    }


def _success(
    artifact: Any,
    **metadata: Any,
) -> dict[str, Any]:
    output_media_type = None
    if isinstance(artifact, dict):
        output_media_type = artifact.get("media_type")

    payload = {
        "success": True,
        "artifact": artifact,
        **metadata,
    }

    if output_media_type is not None:
        payload["output_media_type"] = str(output_media_type)

    return payload


def _binary_response(
    output: Path,
    **metadata: Any,
) -> dict[str, Any]:
    return _success(
        _image_artifact(output),
        **metadata,
    )


# ============================================================
# MASK HELPERS
# ============================================================

def _mask_binary_data(
    binary_dir: Path,
) -> dict[str, str]:
    return {
        path.name: base64.b64encode(
            path.read_bytes()
        ).decode("ascii")
        for path in sorted(
            binary_dir.glob("*.png")
        )
    }


def _mask_json_artifact(
    mask_json: Path,
    binary_dir: Path,
) -> dict[str, Any]:
    with mask_json.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    masks = data.get(
        "masks",
        {},
    )

    if isinstance(masks, dict):
        for name, item in masks.items():
            if not isinstance(item, dict):
                continue

            binary_path = None

            binary_reference = item.get("binary_file")
            if isinstance(binary_reference, str) and binary_reference.strip():
                reference_name = Path(binary_reference).name
                candidates = (
                    mask_json.parent / reference_name,
                    binary_dir / reference_name,
                )
                binary_path = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.is_file()
                    ),
                    None,
                )

            if binary_path is None:
                fallback = binary_dir / f"{name}.png"
                binary_path = fallback if fallback.is_file() else None

            if binary_path is not None:
                item["binary_encoding"] = "base64"

                item["binary_png_base64"] = (
                    base64.b64encode(
                        binary_path.read_bytes()
                    ).decode("ascii")
                )

    return data


def _save_json(
    data: Any,
    path: Path,
) -> None:
    save_json(
        data,
        path,
    )


def _result(
    output: Path,
    source: Optional[Path] = None,
    **extra: Any,
) -> dict[str, Any]:
    if source is not None:
        preserve_metadata(
            source,
            output,
        )

    return result(
        output,
        **extra,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "status": "ok",
    }


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/jobs/{job_id}")
def job_status(
    job_id: str,
) -> dict[str, Any]:
    """Return recorded progress events for a job."""

    job_id = validate_job_id(job_id)

    events = EVENTS.snapshot(
        job_id
    )

    if not events:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    latest = events[-1]

    if latest.get("type") == "error":
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "job_id": job_id,
                "error": latest,
                "events": events,
            },
        )

    if (
        latest.get("type") == "cancelled"
        and latest.get("final")
    ):
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "job_id": job_id,
                "error": latest,
                "events": events,
            },
        )

    return {
        "success": True,
        "job_id": job_id,
        "events": events,
        "latest": latest,
        "finished": (
            latest.get("type")
            in {
                "completed",
                "error",
                "cancelled",
            }
            and latest.get(
                "final",
                False,
            )
        ),
    }


@app.post("/jobs/{job_id}/cancel")
def cancel(
    job_id: str,
) -> dict[str, Any]:
    """Request cooperative cancellation for a running job."""

    job_id = validate_job_id(
        job_id
    )

    if not EVENTS.snapshot(
        job_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    cancel_job(
        job_id
    )

    EVENTS.publish(
        job_id,
        {
            "type": "cancellation-requested",
            "job_id": job_id,
            "stage": "cancelling",
        },
    )

    return {
        "success": True,
        "job_id": job_id,
        "cancel_requested": True,
    }


# ============================================================
# SWAGGER
# ============================================================

@app.get(
    "/swagger",
    include_in_schema=False,
)
def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
    )


# ============================================================
# AI DENOISE
# ============================================================

@job_operation("ai-denoise")
def denoise(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Destination directory",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    device: str = Query(
        "cpu",
        pattern="^(cpu|cuda|auto)$",
    ),
    job_id: Optional[str] = Query(
        None,
        description="Client-generated job ID",
    ),
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "load-input",
        5,
        operation="denoise",
    )

    output = (
        _destination_output(
            path,
            destination,
            "ai-denoise",
        )
        if destination is not None
        else _requested_output(
            path,
            "denoised",
            temporary,
            _job_dir(job_id)
            if temporary
            else None,
            None,
        )
    )

    try:
        image = _read_color(
            path
        )

        publish_progress(
            "prepare-model",
            20,
            operation="denoise",
        )

        try:
            model = ai_denoise.download_model(
                AI_MODEL_CACHE
            )

            image = ai_denoise.ai_denoise(
                image,
                model,
                device=device,
                progress=lambda percent, stage:
                    publish_progress(
                        stage,
                        percent,
                        operation="denoise",
                    ),
                cancel=check_cancelled,
            )

        except Exception as exc:
            ai_denoise.logger.warning(
                "AI denoising failed or unavailable (%s); "
                "using OpenCV fallback",
                exc,
            )

            image = (
                ai_denoise.traditional_fallback(
                    image
                )
            )

        publish_progress(
            "save-output",
            90,
            operation="denoise",
        )

        if not cv2.imwrite(
            str(output),
            image,
        ):
            raise RuntimeError(
                "Could not write denoised image"
            )

        return _result(
            output,
            path,
            device=device,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# ANALYSIS
# ============================================================

@job_operation("analyze")
def analyze(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Optional destination JSON path",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    job_id: Optional[str] = Query(
        None,
        description="Client-generated job ID",
    ),
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "extract-image-data",
        20,
        operation="analyze",
    )

    analysis_path = _requested_output(
        path,
        "analyze",
        temporary,
        _job_dir(job_id)
        if temporary
        else None,
        destination,
        ".json",
    )

    try:
        data = extract_image_data.extract_image_data(
            str(path)
        )

        _save_json(
            data,
            analysis_path,
        )

        publish_progress(
            "write-json",
            90,
            operation="analyze",
            output=str(analysis_path),
        )

        return {
            "analysis": str(
                analysis_path
            ),
            "data": data,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# AUTO ENHANCE - FILE PATH API
# ============================================================

@job_operation("auto-enhance")
def auto_enhance_route(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Destination directory",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    strength: float = Query(
        1.0,
        ge=0.0,
        le=1.0,
    ),
    job_id: Optional[str] = Query(
        None,
        description="Client-generated job ID",
    ),
    image: Optional[bytes] = Body(
        None,
        media_type="application/octet-stream",
    ),
) -> Any:

    binary_input = (
        image is not None
        and len(image) > 0
    )

    publish_progress(
        "load-input",
        5,
        operation="auto-enhance",
    )

    if binary_input:
        path, job = _binary_source(
            image,
            job_id,
        )

        output = (
            job
            / "enhanced.png"
        )

    else:
        if source is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Provide an image request body "
                    "or source query parameter"
                ),
            )

        path = _source(
            source
        )

        output = (
            _destination_output(
                path,
                destination,
                "auto-enhance",
            )
            if destination is not None
            else _requested_output(
                path,
                "enhanced",
                temporary,
                _job_dir(job_id)
                if temporary
                else None,
                None,
            )
        )

    try:
        # --------------------------------------------------------
        # ANALYZE
        # --------------------------------------------------------

        analysis = (
            extract_image_data.extract_image_data(
                str(path)
            )
        )

        analysis_path = output.with_name(
            f"{output.stem}_analysis.json"
        )

        _save_json(
            analysis,
            analysis_path,
        )

        publish_progress(
            "enhance-image",
            55,
            operation="auto-enhance",
        )

        # --------------------------------------------------------
        # ENHANCE
        #
        # IMPORTANT:
        # This intentionally calls the module-level function
        # auto_enhance.enhance().
        # --------------------------------------------------------

        result_image = auto_enhance.enhance(
            _read_color(path),
            analysis,
            strength=strength,
        )

        auto_enhance._save_image(
            result_image,
            output,
        )

        publish_progress(
            "save-output",
            90,
            operation="auto-enhance",
        )

        if binary_input:
            return _binary_response(
                output
            )

        return _result(
            output,
            path,
            analysis=analysis,
            analysis_json=str(
                analysis_path
            ),
            strength=strength,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# APPLY PRESET
# ============================================================

@job_operation("apply-preset")
def preset(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Destination directory",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    preset: Optional[str] = Query(
        None,
        description="Optional XMP preset path",
    ),
    jpeg_quality: int = Query(
        97,
        ge=1,
        le=100,
    ),
    job_id: Optional[str] = Query(
        None,
        description="Client-generated job ID",
    ),
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "load-settings",
        10,
        operation="preset",
    )

    xmp = (
        Path(preset)
        .expanduser()
        .resolve()
        if preset
        else None
    )

    if xmp is not None and not xmp.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Preset not found: {xmp}",
        )

    output = (
        _destination_output(
            path,
            destination,
            "apply-preset",
        )
        if destination is not None
        else _requested_output(
            path,
            "preset",
            temporary,
            _job_dir(job_id)
            if temporary
            else None,
            None,
        )
    )

    try:
        settings = (
            apply_preset.parse_xmp(
                str(xmp)
            )
            if xmp
            else apply_preset.XMPSettings()
        )

        options = apply_preset.PipelineOptions(
            use_exposure=True,
            use_white_balance=True,
            use_tone=True,
            use_point_curve=True,
            use_hsl=True,
            use_color=True,
            use_effects=True,
            use_noise=True,
            use_sharpening=True,
        )

        image, alpha = apply_preset.load_image(
            str(path)
        )

        processed = apply_preset.process_image(
            image,
            alpha,
            settings,
            options,
        )

        publish_progress(
            "save-output",
            90,
            operation="preset",
        )

        apply_preset.save_image(
            str(output),
            processed,
            alpha,
            jpeg_quality=jpeg_quality,
        )

        return _result(
            output,
            path,
            preset=str(xmp)
            if xmp
            else None,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# GEOMETRY
# ============================================================

@job_operation("geometry-correction")
def geometry(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Destination directory",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    mode: str = Query(
        "auto",
        pattern="^(off|auto|level|vertical|horizontal|full)$",
    ),
    rotate: float = Query(0.0),
    vertical_strength: Optional[float] = Query(None),
    horizontal_strength: Optional[float] = Query(None),
    aspect: float = Query(0.0),
    scale: float = Query(
        100.0,
        gt=0.0,
    ),
    x: float = Query(0.0),
    y: float = Query(0.0),
    crop: bool = Query(True),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "load-input",
        10,
        operation="geometry",
    )

    output = (
        _destination_output(
            path,
            destination,
            "geometry-correction",
        )
        if destination is not None
        else _requested_output(
            path,
            "geometry",
            temporary,
            _job_dir(job_id)
            if temporary
            else None,
            None,
        )
    )

    try:
        image = geometry_correction.read_image(
            str(path)
        )

        if mode == "off":
            processed = image
            info = None

        else:
            height, width = image.shape[:2]

            transform, info = (
                geometry_correction.build_transform(
                    image,
                    mode=mode,
                    rotate=rotate,
                    vertical_strength=vertical_strength,
                    horizontal_strength=horizontal_strength,
                    aspect=aspect,
                    scale=scale,
                    x=x,
                    y=y,
                    verbose=False,
                )
            )

            transform, out_w, out_h = (
                geometry_correction.fit_to_canvas(
                    transform,
                    width,
                    height,
                )
            )

            processed = cv2.warpPerspective(
                image,
                transform,
                (out_w, out_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
            )

            if crop:
                valid_mask = (
                    geometry_correction.render_validity_mask(
                        width,
                        height,
                        transform,
                        out_w,
                        out_h,
                    )
                )

                processed = (
                    geometry_correction.crop_using_mask(
                        processed,
                        valid_mask,
                    )
                )

        publish_progress(
            "save-output",
            90,
            operation="geometry",
        )

        geometry_correction.write_image(
            str(output),
            processed,
        )

        return _result(
            output,
            path,
            mode=mode,
            level=getattr(
                info,
                "level",
                0.0,
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# LENS
# ============================================================

@job_operation("lens-correction")
def lens(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Destination directory",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    manual: bool = Query(False),
    k1: float = Query(0.0),
    k2: float = Query(0.0),
    k3: float = Query(0.0),
    p1: float = Query(0.0),
    p2: float = Query(0.0),
    strength: float = Query(1.0),
    center_x: float = Query(0.5),
    center_y: float = Query(0.5),
    focal_scale: float = Query(1.0),
    no_crop: bool = Query(False),
    camera_maker: Optional[str] = Query(None),
    camera_model: Optional[str] = Query(None),
    lens_maker: Optional[str] = Query(None),
    lens_model: Optional[str] = Query(None),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "correct-lens",
        30,
        operation="lens",
    )

    output = (
        _destination_output(
            path,
            destination,
            "lens-correction",
        )
        if destination is not None
        else _requested_output(
            path,
            "lens",
            temporary,
            _job_dir(job_id)
            if temporary
            else None,
            None,
        )
    )

    try:
        options = (
            lens_correction.LensCorrectionOptions(
                use_lensfun=not manual,
                manual=manual,
                auto_crop=not no_crop,
            )
        )

        manual_settings = (
            lens_correction.ManualDistortion(
                k1=k1,
                k2=k2,
                k3=k3,
                p1=p1,
                p2=p2,
                strength=strength,
                center_x=center_x,
                center_y=center_y,
                focal_scale=focal_scale,
            )
        )

        lens_correction.process(
            str(path),
            str(output),
            options,
            manual_settings,
            camera_maker,
            camera_model,
            lens_maker,
            lens_model,
        )

        publish_progress(
            "save-output",
            90,
            operation="lens",
        )

        return _result(
            output,
            path,
            manual=manual,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# MASK EXTRACTION
# ============================================================

def _extract_masks(
    path: Path,
    job: Path,
    device: str = "auto",
) -> tuple[Path, Path]:

    from autophotoeditor.processing import (
        extract_mask_data,
    )

    device = extract_mask_data.choose_device(
        device
    )

    publish_progress(
        "extract-masks",
        25,
        operation="masks",
    )

    check_cancelled()

    mask_json = (
        job
        / f"{path.stem}_masks.json"
    )

    binary_dir = (
        job
        / "binary_masks"
    )

    extract_mask_data.run_pipeline(
        input_path=path,
        output_json=mask_json,
        person_model=str(MODEL_PATH),
        human_model="fashn-ai/fashn-human-parser",
        face_model="jonathandinu/face-parsing",
        device=device,
        person_imgsz=1280,
        person_conf=0.25,
        person_iou=0.45,
        face_threshold=0.5,
        face_padding=0.10,
        visual_path=None,
        visual_dir=None,
        binary_dir=binary_dir,
        soft_dir=None,
        person_masks_dir=None,
        mask_opacity=0.6,
        jpeg_quality=95,
        pretty=True,
        no_rle=False,
        save_person_previews=False,
        feather_sigma=1.5,
    )

    publish_progress(
        "masks-ready",
        85,
        operation="masks",
        output=str(mask_json),
    )

    return mask_json, binary_dir


# ============================================================
# MASK DATA
# ============================================================

@job_operation("mask-data")
def mask_data(
    source: str = Query(...),
    device: str = Query(
        "auto",
        pattern="^(auto|cpu|cuda)$",
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    path = _source(
        source
    )

    job = _job_dir(
        job_id
    )

    publish_progress(
        "prepare-masks",
        5,
        operation="mask-data",
    )

    try:
        mask_json, binary_dir = (
            _extract_masks(
                path,
                job,
                device,
            )
        )

        data = _mask_json_artifact(
            mask_json,
            binary_dir,
        )

        return {
            "masks_json": str(
                mask_json
            ),
            "mask_dir": str(
                binary_dir
            ),
            "mask_json": data,
            "masks": data.get(
                "masks",
                {},
            ),
            "binary_masks": _mask_binary_data(
                binary_dir
            ),
        }

    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Mask extraction failed: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# MASKED ADAPTIVE ENHANCE
# ============================================================

@job_operation("masked-adaptive-enhance")
def masked_enhance(
    source: str = Query(...),
    destination: Optional[str] = None,
    temporary: bool = False,
    masks_json: Optional[str] = None,
    mask_json_data: Optional[str] = None,
    mask_dir: Optional[str] = None,
    masks: Optional[list[str]] = None,
    strength: float = 1.0,
    feather: float = 2.0,
    job_id: Optional[str] = None,
) -> dict[str, Any]:

    path = _source(
        source
    )

    publish_progress(
        "prepare-masked-enhance",
        5,
        operation="masked-enhance",
    )

    job = _job_dir(
        job_id
    )

    output = (
        _destination_output(
            path,
            destination,
            "masked-adaptive-enhance",
        )
        if destination is not None
        else _requested_output(
            path,
            "masked-enhanced",
            temporary,
            job,
            None,
        )
    )

    try:
        # --------------------------------------------------------
        # Decode supplied mask JSON
        # --------------------------------------------------------

        if (
            masks_json is None
            and mask_json_data is not None
        ):
            try:
                provided_masks = json.loads(
                    mask_json_data
                )
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "mask_json_data is not valid JSON"
                    ),
                ) from exc

            provided_json = (
                job
                / "provided_masks.json"
            )

            _save_json(
                provided_masks,
                provided_json,
            )

            masks_json = str(
                provided_json
            )

        # --------------------------------------------------------
        # Generate masks if none supplied
        # --------------------------------------------------------

        if masks_json is None:
            publish_progress(
                "extract-masks",
                10,
                operation="masked-enhance",
            )

            masks_path, generated_dir = (
                _extract_masks(
                    path,
                    job,
                )
            )

            masks_json = str(
                masks_path
            )

            mask_dir = str(
                generated_dir
            )

        json_path = (
            Path(masks_json)
            .expanduser()
            .resolve()
        )

        mask_path = (
            Path(mask_dir)
            .expanduser()
            .resolve()
            if mask_dir
            else None
        )

        masked_adaptive_enhance.process(
            image_path=path,
            json_path=json_path,
            output_path=output,
            mask_dir=mask_path,
            global_strength=strength,
            feather_radius=feather,
            save_mask_results=None,
            mask_names=masks,
            progress=lambda percent, stage:
                publish_progress(
                    stage,
                    percent,
                    operation="masked-enhance",
                ),
        )

        publish_progress(
            "save-output",
            95,
            operation="masked-enhance",
        )

        return _result(
            output,
            path,
            masks_json=str(
                json_path
            ),
        )

    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Mask extraction failed: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PIPELINE
# ============================================================

@job_operation("pipeline")
def pipeline(
    source: str = Query(...),
    destination: Optional[str] = Query(
        None,
        description="Optional final destination",
    ),
    temporary: bool = Query(
        False,
        description="Store output in temporary job directory",
    ),
    stages: str = Query(
        "analyze,auto-enhance"
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:
    """
    Run comma-separated pipeline stages in order.
    """

    current = _source(
        source
    )

    publish_progress(
        "prepare-pipeline",
        5,
        operation="pipeline",
        stages=stages,
    )

    job = _job_dir(
        job_id
    )

    final_destination = _requested_output(
        current,
        "pipeline",
        temporary,
        job,
        destination,
    )

    results: list[dict[str, Any]] = []

    try:
        stage_names = (
            PipelineStage.parse_many(
                stages
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not stage_names:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one pipeline stage "
                "is required"
            ),
        )

    for index, stage in enumerate(
        stage_names
    ):

        check_cancelled()

        publish_progress(
            f"stage-{stage.value}",
            int(
                index * 90
                / len(stage_names)
            ) + 5,
            operation="pipeline",
            stage_index=index + 1,
            stage_count=len(stage_names),
        )

        # --------------------------------------------------------
        # ANALYZE
        # --------------------------------------------------------

        if stage is PipelineStage.ANALYZE:

            results.append(
                analyze(
                    str(current),
                    destination=None,
                    temporary=True,
                    job_id=job_id,
                )
            )

        # --------------------------------------------------------
        # AUTO ENHANCE
        # --------------------------------------------------------

        elif stage is PipelineStage.AUTO_ENHANCE:

            stage_result = (
                auto_enhance_route(
                    str(current),
                    destination=None,
                    temporary=True,
                    strength=1.0,
                    job_id=job_id,
                    image=None,
                )
            )

            results.append(
                stage_result
            )

            current = Path(
                stage_result["output"]
            )

        # --------------------------------------------------------
        # DENOISE
        # --------------------------------------------------------

        elif stage is PipelineStage.DENOISE:

            stage_result = denoise(
                str(current),
                destination=None,
                temporary=True,
                device="cpu",
                job_id=job_id,
            )

            results.append(
                stage_result
            )

            current = Path(
                stage_result["output"]
            )

        # --------------------------------------------------------
        # GEOMETRY
        # --------------------------------------------------------

        elif stage is PipelineStage.GEOMETRY:

            stage_result = geometry(
                str(current),
                destination=None,
                temporary=True,
                mode="auto",
                rotate=0.0,
                vertical_strength=None,
                horizontal_strength=None,
                aspect=0.0,
                scale=100.0,
                x=0.0,
                y=0.0,
                crop=True,
                job_id=job_id,
            )

            results.append(
                stage_result
            )

            current = Path(
                stage_result["output"]
            )

    if (
        current.suffix.lower()
        != final_destination.suffix.lower()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Final destination extension "
                "must match the final image output"
            ),
        )

    if current != final_destination:
        shutil.copy2(
            current,
            final_destination,
        )

    publish_progress(
        "pipeline-complete",
        95,
        operation="pipeline",
        output=str(final_destination),
    )

    return {
        "output": str(
            final_destination
        ),
        "stages": results,
    }


# ============================================================
# JOB ACTION REGISTRY
#
# IMPORTANT:
# This MUST be at module scope.
# ============================================================

_JOB_ACTIONS = {
    "analyze": analyze,
    "auto-enhance": auto_enhance_route,
    "manual-adjust": None,
    "ai-denoise": denoise,
    "apply-preset": preset,
    "geometry-correction": geometry,
    "lens-correction": lens,
    "mask-data": mask_data,
    "masked-adaptive-enhance": masked_enhance,
    "pipeline": pipeline,
}


# ============================================================
# MANUAL ADJUST
#
# Defined before finalizing _JOB_ACTIONS.
# ============================================================

@job_operation("manual-adjust")
def manual_adjust_api(
    payload: dict[str, Any] = Body(...),
    feather: float = Query(
        2.0,
        ge=0.0,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    image_base64 = payload.get(
        "image_base64"
    )

    adjustments = payload.get(
        "adjustments"
    )

    mask_json_data = payload.get(
        "mask_json"
    )

    mask_names_data = payload.get(
        "mask_names"
    )

    if (
        not isinstance(
            image_base64,
            str,
        )
        or not isinstance(
            adjustments,
            dict,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "JSON body must contain "
                "image_base64 and adjustments object"
            ),
        )

    if (
        mask_json_data is not None
        and not isinstance(
            mask_json_data,
            dict,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="mask_json must be an object",
        )

    if mask_names_data is not None:

        if (
            not isinstance(
                mask_names_data,
                list,
            )
            or not all(
                isinstance(name, str)
                and name.strip()
                for name in mask_names_data
            )
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "mask_names must be a list "
                    "of non-empty strings"
                ),
            )

        mask_names = [
            name.strip().lower()
            for name in mask_names_data
        ]

    else:
        mask_names = []

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    try:
        image = base64.b64decode(
            image_base64,
            validate=True,
        )

    except (
        ValueError,
        TypeError,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "image_base64 is not valid base64"
            ),
        )

    source, job = _binary_source(
        image,
        job_id,
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    try:
        source_image = _read_color(
            source
        )

        # ----------------------------------------------------
        # Masked adjustment
        # ----------------------------------------------------

        if (
            mask_json_data is not None
            and mask_names
        ):

            provided_json = (
                job
                / "provided_manual_masks.json"
            )

            _save_json(
                mask_json_data,
                provided_json,
            )

            mask_entries = (
                masked_adaptive_enhance.extract_mask_entries(
                    mask_json_data,
                    provided_json,
                    None,
                )
            )

            selected_masks = [
                mask_entries[name]
                for name in mask_names
                if name in mask_entries
            ]

            if not selected_masks:

                available = (
                    ", ".join(
                        sorted(mask_entries)
                    )
                    or "none"
                )

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "None of the selected masks "
                        f"were found. Available masks: "
                        f"{available}"
                    ),
                )

            combined_mask = None

            for mask_path in selected_masks:

                loaded_mask = (
                    masked_adaptive_enhance.load_mask(
                        mask_path,
                        source_image.shape[1],
                        source_image.shape[0],
                    )
                )

                if loaded_mask is None:
                    continue

                if combined_mask is None:
                    combined_mask = loaded_mask

                else:
                    combined_mask = np.maximum(
                        combined_mask,
                        loaded_mask,
                    )

            if combined_mask is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Selected masks could not "
                        "be loaded"
                    ),
                )

            adjusted, normalized = (
                manual_adjust.apply_masked(
                    source_image,
                    adjustments,
                    combined_mask,
                    feather_radius=feather,
                )
            )

        # ----------------------------------------------------
        # Global adjustment
        # ----------------------------------------------------

        else:

            adjusted, normalized = (
                manual_adjust.apply(
                    source_image,
                    adjustments,
                )
            )

    except manual_adjust.ManualAdjustmentError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        job
        / "manual-adjusted.png"
    )

    if not cv2.imwrite(
        str(output),
        adjusted,
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not write manual "
                "adjustment output"
            ),
        )

    return _success(
        _image_artifact(output),
        adjustments=normalized,
        mask_names=mask_names,
        feather=feather,
    )


# ============================================================
# FIX JOB REGISTRY AFTER MANUAL ADJUST EXISTS
# ============================================================

_JOB_ACTIONS["manual-adjust"] = manual_adjust_api


# ============================================================
# BACKGROUND JOB EXECUTION
# ============================================================

def _run_job_action(
    action: str,
    params: dict[str, Any],
    job_id: str,
) -> None:

    try:
        _JOB_ACTIONS[action](
            **params,
            job_id=job_id,
        )

    except Exception:
        # The job decorator records the failure.
        pass


@app.post(
    "/jobs",
    status_code=202,
)
def submit_job(
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:

    action = payload.get(
        "action"
    )

    params = payload.get(
        "params",
        {},
    )

    if (
        not isinstance(action, str)
        or action not in _JOB_ACTIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported action: {action}"
            ),
        )

    if not isinstance(
        params,
        dict,
    ):
        raise HTTPException(
            status_code=400,
            detail="params must be an object",
        )

    job_id = validate_job_id(
        payload.get(
            "job_id"
        )
        or str(uuid.uuid4())
    )

    cancellation_token(
        job_id
    )

    EVENTS.publish(
        job_id,
        {
            "type": "queued",
            "job_id": job_id,
            "operation": action,
            "stage": "queued",
            "percent": 0,
        },
    )

    background_tasks.add_task(
        _run_job_action,
        action,
        dict(params),
        job_id,
    )

    return {
        "success": True,
        "job_id": job_id,
        "status_url": (
            f"/jobs/{job_id}"
        ),
    }


# ============================================================
# BINARY API - AI DENOISE
# ============================================================

@app.post("/ai-denoise")
@job_operation("ai-denoise")
def denoise_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    device: str = Query(
        "cpu",
        pattern="^(cpu|cuda|auto)$",
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = denoise(
        str(source),
        destination=None,
        temporary=True,
        device=device,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY API - AUTO ENHANCE
# ============================================================

@app.post("/auto-enhance")
@job_operation("auto-enhance")
def auto_enhance_api(
    payload: dict[str, Any] = Body(...),
    strength: float = Query(
        1.0,
        ge=0.0,
        le=1.0,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    image_base64 = payload.get(
        "image_base64"
    )

    analysis = payload.get(
        "analysis"
    )

    if (
        not isinstance(
            image_base64,
            str,
        )
        or not isinstance(
            analysis,
            dict,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "JSON body must contain "
                "image_base64 and analysis object"
            ),
        )

    try:
        image = base64.b64decode(
            image_base64,
            validate=True,
        )

    except (
        ValueError,
        TypeError,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "image_base64 is not valid base64"
            ),
        )

    source, job = _binary_source(
        image,
        job_id,
    )

    output = (
        job
        / "enhanced.png"
    )

    try:
        # --------------------------------------------------------
        # THIS IS THE IMPORTANT LINE.
        #
        # auto_enhance.py MUST expose:
        #
        # def enhance(...)
        #
        # --------------------------------------------------------

        enhanced = auto_enhance.enhance(
            _read_color(source),
            analysis,
            strength=strength,
        )

        auto_enhance._save_image(
            enhanced,
            output,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return _success(
        _image_artifact(output),
        analysis=analysis,
        strength=strength,
    )


# ============================================================
# BINARY API - MANUAL ADJUST
# ============================================================

@app.post("/manual-adjust")
@job_operation("manual-adjust")
def manual_adjust_binary_api(
    payload: dict[str, Any] = Body(...),
    feather: float = Query(
        2.0,
        ge=0.0,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    return manual_adjust_api(
        payload=payload,
        feather=feather,
        job_id=job_id,
    )


# ============================================================
# BINARY API - ANALYSIS DATA
# ============================================================

@app.post("/auto-enhance-data")
@job_operation("analyze")
def auto_enhance_data_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    job_id: Optional[str] = Query(None),
    summary: bool = Query(False),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    analysis = extract_image_data.analyze_image(
        _read_color(source)
    )

    payload = analysis
    if summary:
        payload = {
            "schema_version": analysis.get("schema_version", ""),
            "image": analysis.get("image", analysis.get("metadata", {})),
            "scene": analysis.get("scene", {}),
            "classification": analysis.get("classification", {}),
            "correction_plan": analysis.get("correction_plan", {}),
            "auto_tone": analysis.get("auto_tone", {}),
            "neutral_balance": analysis.get("neutral_balance", {}),
            "white_balance": analysis.get("white_balance", {}),
            "recommendations": analysis.get("recommendations", {}),
        }

    return _success(
        {
            "kind": "json",
            "data": payload,
        }
    )


# ============================================================
# BINARY API - PRESET
# ============================================================

@app.post("/apply-preset")
@job_operation("apply-preset")
def preset_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    preset: Optional[str] = Query(None),
    jpeg_quality: int = Query(
        97,
        ge=1,
        le=100,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = preset(
        str(source),
        destination=None,
        temporary=True,
        preset=preset,
        jpeg_quality=jpeg_quality,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY API - GEOMETRY
# ============================================================

@app.post("/geometry-correction")
@job_operation("geometry-correction")
def geometry_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    mode: str = Query(
        "auto",
        pattern="^(off|auto|level|vertical|horizontal|full)$",
    ),
    rotate: float = Query(0.0),
    vertical_strength: Optional[float] = Query(None),
    horizontal_strength: Optional[float] = Query(None),
    aspect: float = Query(0.0),
    scale: float = Query(
        100.0,
        gt=0.0,
    ),
    x: float = Query(0.0),
    y: float = Query(0.0),
    crop: bool = Query(True),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = geometry(
        str(source),
        destination=None,
        temporary=True,
        mode=mode,
        rotate=rotate,
        vertical_strength=vertical_strength,
        horizontal_strength=horizontal_strength,
        aspect=aspect,
        scale=scale,
        x=x,
        y=y,
        crop=crop,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY API - LENS
# ============================================================

@app.post("/lens-correction")
@job_operation("lens-correction")
def lens_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    manual: bool = Query(False),
    k1: float = Query(0.0),
    k2: float = Query(0.0),
    k3: float = Query(0.0),
    p1: float = Query(0.0),
    p2: float = Query(0.0),
    strength: float = Query(1.0),
    center_x: float = Query(0.5),
    center_y: float = Query(0.5),
    focal_scale: float = Query(1.0),
    no_crop: bool = Query(False),
    camera_maker: Optional[str] = Query(None),
    camera_model: Optional[str] = Query(None),
    lens_maker: Optional[str] = Query(None),
    lens_model: Optional[str] = Query(None),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = lens(
        str(source),
        destination=None,
        temporary=True,
        manual=manual,
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
        strength=strength,
        center_x=center_x,
        center_y=center_y,
        focal_scale=focal_scale,
        no_crop=no_crop,
        camera_maker=camera_maker,
        camera_model=camera_model,
        lens_maker=lens_maker,
        lens_model=lens_model,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY API - MASK DATA
# ============================================================

@app.post("/mask-data")
@job_operation("mask-data")
def mask_data_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    device: str = Query(
        "auto",
        pattern="^(auto|cpu|cuda)$",
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = mask_data(
        str(source),
        device=device,
        job_id=job_id,
    )

    return _success(
        {
            "kind": "json",
            "data": result_data["mask_json"],
            "binary_masks": result_data[
                "binary_masks"
            ],
        },
        masks=result_data["masks"],
    )


# ============================================================
# BINARY API - PIPELINE
# ============================================================

@app.post("/pipeline")
@job_operation("pipeline")
def pipeline_api(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    stages: str = Query(
        "analyze,auto-enhance"
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_path(
        image,
        job_id,
    )

    result_data = pipeline(
        str(source),
        destination=None,
        temporary=True,
        stages=stages,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# MASKED ADAPTIVE ENHANCE - JSON
# ============================================================

@app.post(
    "/masked-adaptive-enhance-json"
)
@job_operation("masked-adaptive-enhance")
def masked_enhance_json_api(
    payload: dict[str, Any] = Body(...),
    masks: Optional[list[str]] = Query(None),
    strength: float = Query(
        1.0,
        ge=0.0,
        le=1.0,
    ),
    feather: float = Query(
        2.0,
        ge=0.0,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    image_base64 = payload.get(
        "image_base64"
    )

    mask_json_data = payload.get(
        "mask_json"
    )

    if (
        not isinstance(
            image_base64,
            str,
        )
        or not isinstance(
            mask_json_data,
            dict,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "JSON body must contain "
                "image_base64 and mask_json object"
            ),
        )

    try:
        image = base64.b64decode(
            image_base64,
            validate=True,
        )

    except (
        ValueError,
        TypeError,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "image_base64 is not valid base64"
            ),
        )

    source, job = _binary_path(
        image,
        job_id,
    )

    provided_json = (
        job
        / "provided_masks.json"
    )

    _save_json(
        mask_json_data,
        provided_json,
    )

    result_data = masked_enhance(
        str(source),
        destination=None,
        temporary=True,
        masks_json=str(
            provided_json
        ),
        mask_dir=str(provided_json.parent),
        masks=masks,
        strength=strength,
        feather=feather,
        job_id=job_id,
    )

    output = Path(
        result_data["output"]
    )

    report_path = output.with_name(
        f"{output.stem}_mask_report.json"
    )

    mask_report = None

    if report_path.is_file():

        try:
            with report_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                mask_report = json.load(
                    handle
                )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            mask_report = None

    return _success(
        _image_artifact(output),
        masks_json=result_data.get(
            "masks_json"
        ),
        strength=strength,
        feather=feather,
        mask_report=mask_report,
    )


# ============================================================
# BINARY DENOISE
# ============================================================

@app.post("/binary/denoise")
@job_operation("denoise")
def binary_denoise(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    device: str = Query(
        "cpu",
        pattern="^(cpu|cuda|auto)$",
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = denoise(
        str(source),
        destination=None,
        temporary=True,
        device=device,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY AUTO ENHANCE
# ============================================================

@app.post("/binary/auto-enhance")
@job_operation("auto-enhance")
def binary_auto_enhance(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    strength: float = Query(
        1.0,
        ge=0.0,
        le=1.0,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = auto_enhance_route(
        str(source),
        destination=None,
        temporary=True,
        strength=strength,
        job_id=job_id,
        image=None,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY PRESET
# ============================================================

@app.post("/binary/preset")
@job_operation("preset")
def binary_preset(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    preset: Optional[str] = Query(None),
    jpeg_quality: int = Query(
        97,
        ge=1,
        le=100,
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = preset(
        str(source),
        destination=None,
        temporary=True,
        preset=preset,
        jpeg_quality=jpeg_quality,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY GEOMETRY
# ============================================================

@app.post("/binary/geometry")
@job_operation("geometry")
def binary_geometry(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    mode: str = Query(
        "auto",
        pattern="^(off|auto|level|vertical|horizontal|full)$",
    ),
    rotate: float = Query(0.0),
    vertical_strength: Optional[float] = Query(None),
    horizontal_strength: Optional[float] = Query(None),
    aspect: float = Query(0.0),
    scale: float = Query(
        100.0,
        gt=0.0,
    ),
    x: float = Query(0.0),
    y: float = Query(0.0),
    crop: bool = Query(True),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = geometry(
        str(source),
        destination=None,
        temporary=True,
        mode=mode,
        rotate=rotate,
        vertical_strength=vertical_strength,
        horizontal_strength=horizontal_strength,
        aspect=aspect,
        scale=scale,
        x=x,
        y=y,
        crop=crop,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY LENS
# ============================================================

@app.post("/binary/lens")
@job_operation("lens")
def binary_lens(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    manual: bool = Query(False),
    k1: float = Query(0.0),
    k2: float = Query(0.0),
    k3: float = Query(0.0),
    p1: float = Query(0.0),
    p2: float = Query(0.0),
    strength: float = Query(1.0),
    center_x: float = Query(0.5),
    center_y: float = Query(0.5),
    focal_scale: float = Query(1.0),
    no_crop: bool = Query(False),
    camera_maker: Optional[str] = Query(None),
    camera_model: Optional[str] = Query(None),
    lens_maker: Optional[str] = Query(None),
    lens_model: Optional[str] = Query(None),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = lens(
        str(source),
        destination=None,
        temporary=True,
        manual=manual,
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
        strength=strength,
        center_x=center_x,
        center_y=center_y,
        focal_scale=focal_scale,
        no_crop=no_crop,
        camera_maker=camera_maker,
        camera_model=camera_model,
        lens_maker=lens_maker,
        lens_model=lens_model,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY MASKED ENHANCE
# ============================================================

@app.post("/binary/masked-enhance")
@job_operation("masked-enhance")
def binary_masked_enhance(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    masks_json: Optional[str] = None,
    mask_dir: Optional[str] = None,
    mask_json_data: Optional[str] = None,
    strength: float = 1.0,
    feather: float = 2.0,
    job_id: Optional[str] = None,
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = masked_enhance(
        str(source),
        destination=None,
        temporary=True,
        masks_json=masks_json,
        mask_dir=mask_dir,
        mask_json_data=mask_json_data,
        strength=strength,
        feather=feather,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# BINARY PIPELINE
# ============================================================

@app.post("/binary/pipeline")
@job_operation("pipeline")
def binary_pipeline(
    image: bytes = Body(
        ...,
        media_type="application/octet-stream",
    ),
    stages: str = Query(
        "analyze,auto-enhance"
    ),
    job_id: Optional[str] = Query(None),
) -> dict[str, Any]:

    source, _ = _binary_source(
        image,
        job_id,
    )

    result_data = pipeline(
        str(source),
        destination=None,
        temporary=True,
        stages=stages,
        job_id=job_id,
    )

    return _binary_response(
        Path(
            result_data["output"]
        )
    )


# ============================================================
# PUBLIC ROUTES
# ============================================================

_PUBLIC_PATHS = {
    "/health",

    # Jobs
    "/jobs",
    "/jobs/{job_id}",
    "/jobs/{job_id}/cancel",

    # Documentation
    "/swagger",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",

    # Main processing APIs
    "/auto-enhance",
    "/manual-adjust",
    "/auto-enhance-data",
    "/ai-denoise",
    "/apply-preset",
    "/geometry-correction",
    "/lens-correction",
    "/masked-adaptive-enhance-json",
    "/mask-data",
    "/pipeline",

    # Binary APIs
    "/binary/denoise",
    "/binary/auto-enhance",
    "/binary/preset",
    "/binary/geometry",
    "/binary/lens",
    "/binary/masked-enhance",
    "/binary/pipeline",
}


# Keep only public processing routes.
app.router.routes[:] = [
    route
    for route in app.router.routes
    if route.path in _PUBLIC_PATHS
]