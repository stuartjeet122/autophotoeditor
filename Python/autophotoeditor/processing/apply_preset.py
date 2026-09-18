#!/usr/bin/env python3
"""
apply_preset.py

Lightroom XMP-inspired image processor.

Usage examples
--------------

Apply ALL supported XMP settings:
    python apply_preset.py preset.xmp input.jpg output.jpg

Apply only HSL:
    python apply_preset.py preset.xmp input.jpg output.jpg --hsl

Apply only tone:
    python apply_preset.py preset.xmp input.jpg output.jpg --tone

Apply HSL + effects:
    python apply_preset.py preset.xmp input.jpg output.jpg --hsl --effects

Override individual tone values:
    python apply_preset.py preset.xmp input.jpg output.jpg --contrast -20

Override HSL:
    python apply_preset.py preset.xmp input.jpg output.jpg \
        --hsl-red "-10,-20,10"

Format for HSL:
    hue,saturation,luminance

Example:
    --hsl-red "-10,-20,10"

If NO processing selector is supplied:
    ALL supported XMP settings are applied.

If at least one processing selector is supplied:
    ONLY selected categories are applied.

Individual parameter arguments automatically enable their category.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autophotoeditor.core.image_io import RAW_EXTENSIONS, decode_color


# ============================================================================
# CONSTANTS
# ============================================================================

HSL_CHANNELS = (
    "red",
    "orange",
    "yellow",
    "green",
    "aqua",
    "blue",
    "purple",
    "magenta",
)

# Approximate hue centers in degrees.
HSL_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 315.0,
}

EPS = 1e-8


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HSLAdjustment:
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0


@dataclass
class ToneSettings:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0


@dataclass
class ColorSettings:
    vibrance: float = 0.0
    saturation: float = 0.0


@dataclass
class WhiteBalanceSettings:
    temperature: float = 0.0
    tint: float = 0.0


@dataclass
class EffectsSettings:
    clarity: float = 0.0
    texture: float = 0.0
    dehaze: float = 0.0


@dataclass
class SharpenSettings:
    amount: float = 0.0
    radius: float = 1.0
    detail: float = 25.0
    masking: float = 0.0


@dataclass
class NoiseSettings:
    luminance: float = 0.0
    luminance_detail: float = 50.0
    color: float = 0.0


@dataclass
class PointCurve:
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class XMPSettings:
    tone: ToneSettings = field(default_factory=ToneSettings)
    wb: WhiteBalanceSettings = field(
        default_factory=WhiteBalanceSettings
    )
    color: ColorSettings = field(default_factory=ColorSettings)
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    sharpen: SharpenSettings = field(default_factory=SharpenSettings)
    noise: NoiseSettings = field(default_factory=NoiseSettings)

    hsl: Dict[str, HSLAdjustment] = field(
        default_factory=lambda: {
            channel: HSLAdjustment()
            for channel in HSL_CHANNELS
        }
    )

    point_curve: PointCurve = field(default_factory=PointCurve)

    camera_profile: Optional[str] = None
    process_version: Optional[str] = None

    tone_curve_name: Optional[str] = None
    tone_curve_name_2012: Optional[str] = None

    unsupported: List[str] = field(default_factory=list)


@dataclass
class PipelineOptions:
    use_exposure: bool = False
    use_white_balance: bool = False
    use_tone: bool = False
    use_point_curve: bool = False
    use_hsl: bool = False
    use_color: bool = False
    use_effects: bool = False
    use_noise: bool = False
    use_sharpening: bool = False

    debug: bool = False


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def finite_image(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def normalize_angle_degrees(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 360.0)


def angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """
    Circular distance between hue arrays and hue center.
    Result is 0..180 degrees.
    """
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return np.zeros_like(x)

    t = np.clip(
        (x - edge0) / (edge1 - edge0),
        0.0,
        1.0,
    )

    return t * t * (3.0 - 2.0 * t)


# ============================================================================
# XMP XML PARSING
# ============================================================================

def local_name(tag: str) -> str:
    """
    Converts:
        {namespace}Exposure2012
    into:
        Exposure2012
    """
    if "}" in tag:
        return tag.rsplit("}", 1)[1]

    return tag


def parse_xmp(path: str) -> XMPSettings:
    """
    Parse Lightroom XMP using XML instead of regex.

    This handles namespaced XML and nested RDF structures.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"XMP preset not found: {path}"
        )

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Could not parse XMP XML: {exc}"
        ) from exc

    settings = XMPSettings()

    attributes: Dict[str, str] = {}

    for element in root.iter():
        for key, value in element.attrib.items():
            name = local_name(key)
            attributes[name] = value

    # ------------------------------------------------------------------------
    # Tone
    # ------------------------------------------------------------------------

    settings.tone.exposure = safe_float(
        attributes.get("Exposure2012")
    )

    settings.tone.contrast = safe_float(
        attributes.get("Contrast2012")
    )

    settings.tone.highlights = safe_float(
        attributes.get("Highlights2012")
    )

    settings.tone.shadows = safe_float(
        attributes.get("Shadows2012")
    )

    settings.tone.whites = safe_float(
        attributes.get("Whites2012")
    )

    settings.tone.blacks = safe_float(
        attributes.get("Blacks2012")
    )

    # ------------------------------------------------------------------------
    # White balance
    # ------------------------------------------------------------------------

    settings.wb.temperature = safe_float(
        attributes.get("IncrementalTemperature")
    )

    settings.wb.tint = safe_float(
        attributes.get("IncrementalTint")
    )

    # ------------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------------

    settings.color.vibrance = safe_float(
        attributes.get("Vibrance")
    )

    settings.color.saturation = safe_float(
        attributes.get("Saturation")
    )

    # ------------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------------

    # Lightroom XMP normally uses "Dehaze".
    settings.effects.dehaze = safe_float(
        attributes.get("Dehaze")
    )

    settings.effects.clarity = safe_float(
        attributes.get("Clarity2012")
        if "Clarity2012" in attributes
        else attributes.get("Clarity")
    )

    settings.effects.texture = safe_float(
        attributes.get("Texture")
    )

    # ------------------------------------------------------------------------
    # Sharpen
    # ------------------------------------------------------------------------

    settings.sharpen.amount = safe_float(
        attributes.get("Sharpness")
    )

    settings.sharpen.radius = safe_float(
        attributes.get("SharpenRadius"),
        1.0,
    )

    settings.sharpen.detail = safe_float(
        attributes.get("SharpenDetail"),
        25.0,
    )

    settings.sharpen.masking = safe_float(
        attributes.get("SharpenEdgeMasking")
    )

    # ------------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------------

    settings.noise.luminance = safe_float(
        attributes.get("LuminanceSmoothing")
    )

    settings.noise.luminance_detail = safe_float(
        attributes.get("LuminanceNoiseReductionDetail"),
        50.0,
    )

    settings.noise.color = safe_float(
        attributes.get("ColorNoiseReduction")
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    settings.camera_profile = attributes.get(
        "CameraProfile"
    )

    settings.process_version = attributes.get(
        "ProcessVersion"
    )

    settings.tone_curve_name = attributes.get(
        "ToneCurveName"
    )

    settings.tone_curve_name_2012 = attributes.get(
        "ToneCurveName2012"
    )

    # ------------------------------------------------------------------------
    # HSL
    # ------------------------------------------------------------------------

    for channel in HSL_CHANNELS:
        adobe_name = channel.capitalize()

        settings.hsl[channel] = HSLAdjustment(
            hue=safe_float(
                attributes.get(
                    f"HueAdjustment{adobe_name}"
                )
            ),
            saturation=safe_float(
                attributes.get(
                    f"SaturationAdjustment{adobe_name}"
                )
            ),
            luminance=safe_float(
                attributes.get(
                    f"LuminanceAdjustment{adobe_name}"
                )
            ),
        )

    # ------------------------------------------------------------------------
    # Point curves
    # ------------------------------------------------------------------------

    settings.point_curve.points = parse_point_curve(root)

    # ------------------------------------------------------------------------
    # Unsupported Adobe features
    # ------------------------------------------------------------------------

    known_unsupported = [
        "CameraProfile",
        "LensProfileName",
        "LensProfileDigest",
        "LensProfileSetup",
        "AutoLateralCA",
        "RetouchArea",
        "MaskGroupBasedCorrections",
        "PaintBasedCorrections",
        "GradientBasedCorrections",
        "RadialBasedCorrections",
        "FaceBasedCorrections",
        "SubjectMask",
        "SkyMask",
    ]

    for feature in known_unsupported:
        if feature in attributes:
            settings.unsupported.append(feature)

    return settings


# ============================================================================
# POINT CURVE PARSING
# ============================================================================

def parse_curve_points_text(text: str) -> List[Tuple[float, float]]:
    """
    Parse strings such as:

        0, 0 32, 20 128, 128 255, 255

    or:

        0,0;32,20;128,128;255,255
    """

    if not text:
        return []

    # Normalize common separators.
    normalized = (
        text
        .replace(";", " ")
        .replace("(", "")
        .replace(")", "")
    )

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        normalized,
    )

    values = [safe_float(v) for v in numbers]

    points = []

    for i in range(0, len(values) - 1, 2):
        x = values[i]
        y = values[i + 1]

        points.append((x, y))

    return points


def parse_point_curve(root: ET.Element) -> List[Tuple[float, float]]:
    """
    Lightroom stores point curves in RDF containers.

    Typical values can appear as:

        crs:ToneCurvePV2012="0, 0, 32, 22, ..."

    or as nested RDF structures.

    This parser searches both attribute values and nested element text.
    """

    curve_names = {
        "ToneCurvePV2012",
        "ToneCurve",
        "ToneCurvePV2012Red",
        "ToneCurvePV2012Green",
        "ToneCurvePV2012Blue",
    }

    # First search attributes.
    for element in root.iter():

        for key, value in element.attrib.items():
            name = local_name(key)

            if name in curve_names:
                points = parse_curve_points_text(value)

                if len(points) >= 2:
                    return normalize_curve_points(points)

    # Then search nested text.
    for element in root.iter():
        name = local_name(element.tag)

        if name in curve_names:
            text = "".join(
                element.itertext()
            )

            points = parse_curve_points_text(text)

            if len(points) >= 2:
                return normalize_curve_points(points)

    return []


def normalize_curve_points(
    points: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:

    normalized = []

    for x, y in points:

        if x > 1.0 or y > 1.0:
            x /= 255.0
            y /= 255.0

        normalized.append(
            (
                float(np.clip(x, 0.0, 1.0)),
                float(np.clip(y, 0.0, 1.0)),
            )
        )

    normalized.sort(key=lambda p: p[0])

    # Remove duplicate x coordinates.
    result = []

    last_x = None

    for x, y in normalized:
        if last_x is not None and abs(x - last_x) < EPS:
            result[-1] = (x, y)
        else:
            result.append((x, y))

        last_x = x

    return result


# ============================================================================
# COLOR SPACE CONVERSIONS
# ============================================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)

    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4)
        - 0.055,
    ).astype(np.float32)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    RGB sRGB -> OKLab.
    """

    linear = srgb_to_linear(rgb)

    matrix1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float32,
    )

    lms = linear @ matrix1.T

    lms = np.cbrt(
        np.maximum(lms, 0.0)
    )

    matrix2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float32,
    )

    return (lms @ matrix2.T).astype(np.float32)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:

    matrix1 = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ],
        dtype=np.float32,
    )

    lms = lab @ matrix1.T
    lms = lms ** 3

    matrix2 = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ],
        dtype=np.float32,
    )

    linear = lms @ matrix2.T

    return clamp01(
        linear_to_srgb(linear)
    )


def rgb_to_oklch(
    rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    lab = rgb_to_oklab(rgb)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    C = np.sqrt(
        a * a + b * b
    )

    H = np.degrees(
        np.arctan2(b, a)
    )

    H = np.mod(H, 360.0)

    return L, C, H


def oklch_to_rgb(
    L: np.ndarray,
    C: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    radians = np.radians(H)

    a = C * np.cos(radians)
    b = C * np.sin(radians)

    lab = np.stack(
        [L, a, b],
        axis=-1,
    )

    return oklab_to_rgb(lab)


# ============================================================================
# IMAGE INPUT / OUTPUT
# ============================================================================

def load_image(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns RGB float32 [0,1] and optional alpha.
    """

    if os.path.splitext(path)[1].lower() in RAW_EXTENSIONS:
        image = cv2.cvtColor(decode_color(Path(path)), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    if image.ndim == 2:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

        rgb = (
            rgb.astype(np.float32) / 255.0
        )

        return rgb, None

    if image.shape[2] == 4:
        bgr = image[..., :3]

        alpha = (
            image[..., 3].astype(np.float32)
            / 255.0
        )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        return (
            rgb.astype(np.float32) / 255.0,
            alpha,
        )

    bgr = image[..., :3]

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        rgb.astype(np.float32) / 255.0,
        None,
    )


def save_image(
    path: str,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    jpeg_quality: int = 97,
):
    rgb = finite_image(rgb)

    rgb8 = np.round(
        rgb * 255.0
    ).astype(np.uint8)

    ext = os.path.splitext(path)[1].lower()

    if alpha is not None:

        alpha8 = np.round(
            np.clip(alpha, 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        rgba = np.dstack(
            [rgb8, alpha8]
        )

        image = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGRA,
        )

    else:

        image = cv2.cvtColor(
            rgb8,
            cv2.COLOR_RGB2BGR,
        )

    params = []

    if ext in {".jpg", ".jpeg"}:
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(np.clip(jpeg_quality, 1, 100)),
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ]

    elif ext == ".png":
        params = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    ok = cv2.imwrite(
        path,
        image,
        params,
    )

    if not ok:
        raise RuntimeError(
            f"Could not write image: {path}"
        )


# ============================================================================
# DEBUG OUTPUT
# ============================================================================

def save_debug_stage(
    image: np.ndarray,
    alpha: Optional[np.ndarray],
    debug_dir: str,
    name: str,
):
    os.makedirs(
        debug_dir,
        exist_ok=True,
    )

    path = os.path.join(
        debug_dir,
        f"{name}.png",
    )

    save_image(
        path,
        image,
        alpha,
    )

    print(
        f"[DEBUG] {path}"
    )


def image_statistics(
    image: np.ndarray,
) -> str:

    return (
        f"min={float(image.min()):.4f}, "
        f"max={float(image.max()):.4f}, "
        f"mean={float(image.mean()):.4f}"
    )


# ============================================================================
# EXPOSURE
# ============================================================================

def apply_exposure(
    image: np.ndarray,
    exposure: float,
) -> np.ndarray:

    if abs(exposure) < EPS:
        return image

    # Exposure is measured in stops.
    multiplier = 2.0 ** exposure

    linear = srgb_to_linear(image)

    linear *= multiplier

    return clamp01(
        linear_to_srgb(linear)
    )


# ============================================================================
# WHITE BALANCE
# ============================================================================

def apply_white_balance(
    image: np.ndarray,
    temperature: float,
    tint: float,
) -> np.ndarray:

    if abs(temperature) < EPS and abs(tint) < EPS:
        return image

    result = image.copy()

    # Conservative approximation.
    #
    # Lightroom's temperature scale is not a simple RGB multiplier.
    # These small coefficients prevent extreme color casts.

    temp_strength = temperature / 100.0
    tint_strength = tint / 100.0

    result[..., 0] *= (
        1.0 + temp_strength * 0.10
    )

    result[..., 2] *= (
        1.0 - temp_strength * 0.10
    )

    result[..., 1] *= (
        1.0 + tint_strength * 0.04
    )

    result[..., 0] *= (
        1.0 - tint_strength * 0.02
    )

    result[..., 2] *= (
        1.0 - tint_strength * 0.02
    )

    return clamp01(result)


# ============================================================================
# LUMINANCE
# ============================================================================

def perceptual_luminance(
    image: np.ndarray,
) -> np.ndarray:

    # Rec.709-like luminance.
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)


# ============================================================================
# TONE
# ============================================================================

def apply_tone(
    image: np.ndarray,
    tone: ToneSettings,
) -> np.ndarray:

    result = image.copy()

    # ------------------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------------------

    if abs(tone.contrast) > EPS:

        strength = tone.contrast / 100.0

        # Smooth contrast approximation.
        factor = 1.0 + strength * 1.5

        result = (
            (result - 0.5) * factor
            + 0.5
        )

    # ------------------------------------------------------------------------
    # Highlights / Shadows
    # ------------------------------------------------------------------------

    luminance = perceptual_luminance(
        result
    )

    if abs(tone.highlights) > EPS:

        amount = tone.highlights / 100.0

        # Highlight mask.
        mask = smoothstep(
            0.45,
            0.95,
            luminance,
        )

        # Preserve shadows.
        influence = mask * mask

        result += (
            amount
            * 0.22
            * influence[..., None]
        )

    if abs(tone.shadows) > EPS:

        amount = tone.shadows / 100.0

        # Shadow mask.
        mask = 1.0 - smoothstep(
            0.05,
            0.60,
            luminance,
        )

        influence = mask * mask

        result += (
            amount
            * 0.22
            * influence[..., None]
        )

    # ------------------------------------------------------------------------
    # Whites
    # ------------------------------------------------------------------------

    if abs(tone.whites) > EPS:

        amount = tone.whites / 100.0

        mask = smoothstep(
            0.72,
            1.00,
            luminance,
        )

        result += (
            amount
            * 0.18
            * mask[..., None]
        )

    # ------------------------------------------------------------------------
    # Blacks
    # ------------------------------------------------------------------------

    if abs(tone.blacks) > EPS:

        amount = tone.blacks / 100.0

        mask = 1.0 - smoothstep(
            0.00,
            0.30,
            luminance,
        )

        result += (
            amount
            * 0.18
            * mask[..., None]
        )

    return clamp01(result)


# ============================================================================
# POINT CURVE
# ============================================================================

def apply_point_curve(
    image: np.ndarray,
    curve: PointCurve,
) -> np.ndarray:

    if len(curve.points) < 2:
        return image

    points = curve.points

    xs = np.array(
        [p[0] for p in points],
        dtype=np.float32,
    )

    ys = np.array(
        [p[1] for p in points],
        dtype=np.float32,
    )

    # Guarantee endpoints.
    if xs[0] > 0.0:
        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, ys[0])

    if xs[-1] < 1.0:
        xs = np.append(xs, 1.0)
        ys = np.append(ys, ys[-1])

    result = np.interp(
        image,
        xs,
        ys,
    )

    return clamp01(
        result.astype(np.float32)
    )


# ============================================================================
# HSL
# ============================================================================

def parse_hsl_override(
    value: str,
) -> HSLAdjustment:

    parts = [
        x.strip()
        for x in value.split(",")
    ]

    if len(parts) != 3:
        raise ValueError(
            "HSL override must be "
            "hue,saturation,luminance"
        )

    return HSLAdjustment(
        hue=safe_float(parts[0]),
        saturation=safe_float(parts[1]),
        luminance=safe_float(parts[2]),
    )


def hsl_channel_mask(
    hue: np.ndarray,
    center: float,
) -> np.ndarray:

    distance = angular_distance(
        hue,
        center,
    )

    # Lightroom's channel ranges overlap.
    # We use a smooth bell-like response rather
    # than a hard triangular edge.

    width = 38.0

    mask = np.clip(
        1.0 - distance / width,
        0.0,
        1.0,
    )

    # Smooth the transition.
    mask = mask * mask * (
        3.0 - 2.0 * mask
    )

    return mask.astype(np.float32)


def apply_hsl(
    image: np.ndarray,
    hsl: Dict[str, HSLAdjustment],
) -> np.ndarray:

    if not hsl:
        return image

    result = image.copy()

    L, C, H = rgb_to_oklch(
        result
    )

    original_H = H.copy()
    original_C = C.copy()

    # Saturation threshold.
    #
    # HSL adjustments should have little effect on
    # near-gray pixels.
    saturation_weight = np.clip(
        original_C / 0.18,
        0.0,
        1.0,
    )

    total_hue_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_sat_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_lum_shift = np.zeros_like(
        H,
        dtype=np.float32,
    )

    total_weight = np.zeros_like(
        H,
        dtype=np.float32,
    )

    # IMPORTANT:
    # Every mask is calculated from ORIGINAL hue.
    #
    # This prevents:
    #
    # Green -> Yellow
    # then the modified Yellow -> Orange
    #
    # cross-talk that causes residue/artifacts.
    for channel in HSL_CHANNELS:

        adjustment = hsl.get(
            channel,
            HSLAdjustment(),
        )

        if (
            abs(adjustment.hue) < EPS
            and abs(adjustment.saturation) < EPS
            and abs(adjustment.luminance) < EPS
        ):
            continue

        center = HSL_CENTERS[channel]

        mask = hsl_channel_mask(
            original_H,
            center,
        )

        mask *= saturation_weight

        total_weight += mask

        # --------------------------------------------------------------------
        # Hue
        # --------------------------------------------------------------------

        if abs(adjustment.hue) > EPS:

            # Lightroom HSL hue is approximately -100..+100.
            #
            # It is NOT a literal degree value.
            #
            # Use a conservative maximum shift.
            hue_shift = (
                adjustment.hue
                * 0.45
            )

            total_hue_shift += (
                mask * hue_shift
            )

        # --------------------------------------------------------------------
        # Saturation
        # --------------------------------------------------------------------

        if abs(adjustment.saturation) > EPS:

            # Scale chroma rather than raw RGB saturation.
            saturation_change = (
                adjustment.saturation
                / 100.0
            )

            total_sat_shift += (
                mask * saturation_change
            )

        # --------------------------------------------------------------------
        # Luminance
        # --------------------------------------------------------------------

        if abs(adjustment.luminance) > EPS:

            luminance_change = (
                adjustment.luminance
                / 100.0
            )

            total_lum_shift += (
                mask * luminance_change
            )

    # Normalize overlapping channels.
    normalization = np.maximum(
        total_weight,
        1.0,
    )

    hue_shift = (
        total_hue_shift
        / normalization
    )

    sat_shift = (
        total_sat_shift
        / normalization
    )

    lum_shift = (
        total_lum_shift
        / normalization
    )

    # ------------------------------------------------------------------------
    # Apply hue
    # ------------------------------------------------------------------------

    H = normalize_angle_degrees(
        original_H + hue_shift
    )

    # ------------------------------------------------------------------------
    # Apply saturation
    # ------------------------------------------------------------------------

    # Chroma scale.
    C = C * (
        1.0 + sat_shift * 0.65
    )

    C = np.maximum(
        C,
        0.0,
    )

    # ------------------------------------------------------------------------
    # Apply luminance
    # ------------------------------------------------------------------------

    L = L + (
        lum_shift * 0.16
    )

    L = np.clip(
        L,
        0.0,
        1.0,
    )

    result = oklch_to_rgb(
        L,
        C,
        H,
    )

    return clamp01(result)


# ============================================================================
# GLOBAL COLOR
# ============================================================================

def apply_color(
    image: np.ndarray,
    color: ColorSettings,
) -> np.ndarray:

    result = image.copy()

    # ------------------------------------------------------------------------
    # Global saturation
    # ------------------------------------------------------------------------

    if abs(color.saturation) > EPS:

        L, C, H = rgb_to_oklch(
            result
        )

        amount = color.saturation / 100.0

        C *= (
            1.0 + amount * 0.70
        )

        C = np.maximum(
            C,
            0.0,
        )

        result = oklch_to_rgb(
            L,
            C,
            H,
        )

    # ------------------------------------------------------------------------
    # Vibrance
    # ------------------------------------------------------------------------

    if abs(color.vibrance) > EPS:

        L, C, H = rgb_to_oklch(
            result
        )

        amount = color.vibrance / 100.0

        # Stronger on moderately saturated pixels.
        #
        # Very saturated colors receive less effect.
        saturation_factor = np.clip(
            1.0 - C / 0.30,
            0.0,
            1.0,
        )

        # Do not boost almost-gray pixels aggressively.
        lower_bound = smoothstep(
            0.015,
            0.05,
            C,
        )

        weight = (
            saturation_factor
            * lower_bound
        )

        C *= (
            1.0
            + amount
            * 0.75
            * weight
        )

        C = np.maximum(
            C,
            0.0,
        )

        result = oklch_to_rgb(
            L,
            C,
            H,
        )

    return clamp01(result)


# ============================================================================
# CLARITY
# ============================================================================

def apply_clarity(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    # Work in luminance to reduce color halos.
    L = perceptual_luminance(
        image
    )

    L8 = np.round(
        np.clip(L, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    sigma = 2.0

    blurred = cv2.GaussianBlur(
        L8,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    ).astype(np.float32) / 255.0

    detail = L - blurred

    enhanced = (
        L
        + detail
        * strength
        * 0.9
    )

    delta = enhanced - L

    result = image + delta[..., None]

    return clamp01(result)


# ============================================================================
# TEXTURE
# ============================================================================

def apply_texture(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    L = perceptual_luminance(
        image
    )

    # Small-scale detail.
    blur = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=0.8,
        sigmaY=0.8,
    )

    detail = L - blur

    result = (
        image
        + detail[..., None]
        * strength
        * 0.45
    )

    return clamp01(result)


# ============================================================================
# DEHAZE
# ============================================================================

def apply_dehaze(
    image: np.ndarray,
    amount: float,
) -> np.ndarray:

    if abs(amount) < EPS:
        return image

    strength = amount / 100.0

    result = image.copy()

    # Conservative approximation:
    #
    # Positive dehaze:
    #   increase local contrast
    #   slightly deepen blacks
    #
    # Negative dehaze:
    #   reduce local contrast
    #   lift blacks

    L = perceptual_luminance(
        result
    )

    blurred = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=8.0,
        sigmaY=8.0,
    )

    local = L - blurred

    result += (
        local[..., None]
        * strength
        * 0.55
    )

    result *= (
        1.0
        - strength * 0.06
    )

    result += (
        strength * 0.025
    )

    return clamp01(result)


def apply_effects(
    image: np.ndarray,
    effects: EffectsSettings,
) -> np.ndarray:

    result = image

    if abs(effects.texture) > EPS:
        result = apply_texture(
            result,
            effects.texture,
        )

    if abs(effects.clarity) > EPS:
        result = apply_clarity(
            result,
            effects.clarity,
        )

    if abs(effects.dehaze) > EPS:
        result = apply_dehaze(
            result,
            effects.dehaze,
        )

    return clamp01(result)


# ============================================================================
# NOISE REDUCTION
# ============================================================================

def apply_noise_reduction(
    image: np.ndarray,
    settings: NoiseSettings,
) -> np.ndarray:

    if (
        abs(settings.luminance) < EPS
        and abs(settings.color) < EPS
    ):
        return image

    result = image.copy()

    strength = max(
        settings.luminance,
        settings.color,
    )

    if strength <= 0:
        return result

    # OpenCV bilateral filter gives a reasonably safe
    # photographic noise reduction without destroying
    # edges as aggressively as Gaussian blur.

    diameter = 5

    sigma_color = (
        10.0
        + strength * 0.7
    )

    sigma_space = 2.0

    bgr = cv2.cvtColor(
        np.round(
            result * 255.0
        ).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    filtered = cv2.bilateralFilter(
        bgr,
        diameter,
        sigma_color,
        sigma_space,
    )

    filtered = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    blend = np.clip(
        strength / 100.0,
        0.0,
        1.0,
    )

    result = (
        result * (1.0 - blend)
        + filtered * blend
    )

    return clamp01(result)


# ============================================================================
# SHARPENING
# ============================================================================

def apply_sharpening(
    image: np.ndarray,
    settings: SharpenSettings,
) -> np.ndarray:

    if settings.amount <= 0:
        return image

    amount = np.clip(
        settings.amount / 100.0,
        0.0,
        3.0,
    )

    radius = np.clip(
        settings.radius,
        0.2,
        5.0,
    )

    L = perceptual_luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
    )

    detail = L - blurred

    # Masking:
    # Lightroom masking suppresses sharpening in
    # relatively flat areas.

    gradient_x = cv2.Sobel(
        L,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        L,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = np.sqrt(
        gradient_x * gradient_x
        + gradient_y * gradient_y
    )

    gradient /= (
        np.max(gradient)
        + EPS
    )

    masking = np.clip(
        settings.masking / 100.0,
        0.0,
        1.0,
    )

    mask = (
        gradient
        * (1.0 - masking)
        + masking
    )

    sharpened_luminance = (
        L
        + detail
        * amount
        * mask
    )

    delta = (
        sharpened_luminance
        - L
    )

    result = (
        image
        + delta[..., None]
    )

    return clamp01(result)


# ============================================================================
# CLI
# ============================================================================

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


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

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