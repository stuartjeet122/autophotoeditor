from __future__ import annotations

from .core import *

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


