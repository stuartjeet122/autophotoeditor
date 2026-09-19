"""
Production-Grade AI Denoising using SwinIR (ONNX) with Tiling.

Downloads a verified, state-of-the-art ONNX model from Hugging Face.
Uses advanced patch-tiling with gradient blending to support ANY image size
without seams or memory crashes.

Defaults to CPU. Use --device cuda for Nvidia GPU.
Automatically falls back to OpenCV denoising if AI processing fails.
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import sys
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from autophotoeditor.core.image_io import decode_color

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger("ai_denoise")

_SESSION_CACHE: dict[tuple[str, str], "ort.InferenceSession"] = {}
_SESSION_LOCK = threading.Lock()

# Verified, working Hugging Face ONNX model for color image denoising
MODEL_URL = "https://huggingface.co/Heliosoph/swinir-onnx/resolve/main/swinir_denoising_color_25.onnx"
MODEL_NAME = "swinir_denoising_color_25.onnx"
PATCH_SIZE = 128  # Model expects 128x128
OVERLAP = 16      # Enough overlap for blending with fewer model calls

class AIDenoiseError(Exception):
    pass

def download_model(model_dir: Path) -> Path:
    model_path = model_dir / MODEL_NAME
    if model_path.exists():
        logger.info("Model already exists at: %s", model_path)
        return model_path

    logger.info("Model not found. Downloading AI model from Hugging Face (~55 MB)...")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rDownloading: {percent}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, str(model_path), reporthook=reporthook)
        print()  # Newline after progress bar
        logger.info("Model downloaded successfully.")
        return model_path
    except Exception as e:
        raise AIDenoiseError(f"Failed to download model: {e}")

def ai_denoise(
    image: np.ndarray,
    model_path: Path,
    device: str = "cpu",
    progress: Callable[[int, str], None] | None = None,
    cancel: Callable[[], None] | None = None,
) -> np.ndarray:
    if not ONNX_AVAILABLE:
        raise AIDenoiseError("onnxruntime is not installed. Run: pip install onnxruntime")

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "auto":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    cache_key = (str(model_path.resolve()), providers[0])
    with _SESSION_LOCK:
        session = _SESSION_CACHE.get(cache_key)
        if session is None:
            logger.info(
                "Loading ONNX model (device: %s)...",
                "CUDA" if "CUDA" in str(providers[0]) else "CPU",
            )
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
            _SESSION_CACHE[cache_key] = session
        else:
            logger.info("Reusing cached ONNX model session (device: %s)...", providers[0])
    input_name = session.get_inputs()[0].name

    h, w = image.shape[:2]
    stride = PATCH_SIZE - OVERLAP
    
    # Calculate padded dimensions to ensure full coverage with tiling
    num_y = max(1, math.ceil(h / stride))
    num_x = max(1, math.ceil(w / stride))
    
    target_h = num_y * stride + OVERLAP
    target_w = num_x * stride + OVERLAP
    
    pad_h = target_h - h
    pad_w = target_w - w
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    
    out_img = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_sum = np.zeros((target_h, target_w), dtype=np.float32)
    
    # Create soft blending mask to avoid seams at patch boundaries
    mask = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i in range(OVERLAP):
        weight = (i + 1) / OVERLAP
        mask[i, :] *= weight
        mask[PATCH_SIZE - 1 - i, :] *= weight
        mask[:, i] *= weight
        mask[:, PATCH_SIZE - 1 - i] *= weight
        
    total_patches = num_y * num_x
    processed = 0
    
    logger.info("Processing %d patches (this may take a moment)...", total_patches)
    
    for y in range(0, num_y * stride, stride):
        for x in range(0, num_x * stride, stride):
            if cancel is not None:
                cancel()
            patch = padded_img[y : y+PATCH_SIZE, x : x+PATCH_SIZE]
            
            # Preprocess: BGR -> RGB, normalize [0,1], transpose NCHW
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            rgb_float = rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(rgb_float, (2, 0, 1))[None, ...]
            
            # Inference
            outputs = session.run(None, {input_name: input_tensor})
            denoised_tensor = outputs[0][0]
            
            # Postprocess
            denoised_patch = np.transpose(denoised_tensor, (1, 2, 0))
            denoised_patch = np.clip(denoised_patch * 255.0, 0, 255).astype(np.float32)
            denoised_patch = cv2.cvtColor(denoised_patch, cv2.COLOR_RGB2BGR)
            
            # Blend into output
            out_img[y : y+PATCH_SIZE, x : x+PATCH_SIZE] += denoised_patch * mask[:, :, None]
            weight_sum[y : y+PATCH_SIZE, x : x+PATCH_SIZE] += mask
            
            processed += 1
            if processed % 10 == 0 or processed == total_patches:
                percent = int(processed * 100 / total_patches)
                if progress is not None:
                    progress(percent, "denoise-patches")
                print(f"\rProgress: {percent}% ({processed}/{total_patches} patches)", end="", flush=True)
                
    print()  # Newline
    
    # Normalize and crop back to original size
    weight_sum = np.maximum(weight_sum, 1e-5)
    out_img /= weight_sum[:, :, None]
    final_img = np.clip(out_img[:h, :w, :], 0, 255).astype(np.uint8)
    
    return final_img

def traditional_fallback(image: np.ndarray) -> np.ndarray:
    logger.info("Applying traditional OpenCV denoising fallback...")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_out = cv2.fastNlMeansDenoising(L, None, h=12, templateWindowSize=7, searchWindowSize=21)
    a_out = cv2.fastNlMeansDenoising(a, None, h=25, templateWindowSize=7, searchWindowSize=21)
    b_out = cv2.fastNlMeansDenoising(b, None, h=25, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(cv2.merge([L_out, a_out, b_out]), cv2.COLOR_LAB2BGR)

def main() -> int:
    parser = argparse.ArgumentParser(description="Production-Grade AI Denoising (SwinIR + Tiling).")
    parser.add_argument("input_image", type=Path, help="Input image path")
    parser.add_argument("output_image", type=Path, help="Output image path")
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda", "auto"], 
        default="cpu", 
        help="Device to run inference on. Default is 'cpu'. Use 'cuda' for Nvidia GPU."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if not args.input_image.exists():
        logger.error("Input image not found: %s", args.input_image)
        return 2

    try:
        img = decode_color(args.input_image)
    except RuntimeError as exc:
        logger.error("Failed to read image: %s", exc)
        return 1

    denoised_img = None
    
    try:
        # 1. Download model if it doesn't exist
        cache_dir = Path.home() / ".cache" / "ai_denoise"
        model_path = download_model(cache_dir)
        
        # 2. Attempt AI denoising
        logger.info("Starting AI denoising process...")
        denoised_img = ai_denoise(img, model_path, device=args.device)
        logger.info("AI denoising completed successfully.")
        
    except Exception as e:
        # 3. If AI fails for ANY reason, seamlessly continue with OpenCV
        logger.warning("AI denoising failed or unavailable (%s). Continuing with OpenCV fallback...", e)
        denoised_img = traditional_fallback(img)

    if denoised_img is None:
        logger.error("All denoising methods failed.")
        return 1

    # Save the result
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    ext = args.output_image.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 98]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        
    cv2.imwrite(str(args.output_image), denoised_img, params)
    logger.info("Successfully saved denoised image to: %s", args.output_image)
    return 0

if __name__ == "__main__":
    sys.exit(main())