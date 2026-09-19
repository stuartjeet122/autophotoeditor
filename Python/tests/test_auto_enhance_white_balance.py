import numpy as np

from autophotoeditor.processing.auto_enhance import EnhancementConfig, enhance


def _make_analysis() -> dict:
    return {
        "white_balance": {
            "correction_gains_rgb": [1.35, 1.0, 0.82],
            "neutral_pixel_coverage_percentage": 10.0,
            "estimated_temperature_cast": {"confidence": 0.8},
        },
        "recommendations": {
            "exposure_ev": {"value": 0.0},
            "highlights": 0.0,
            "shadows": 0.0,
        },
    }


def test_enhance_removes_warm_cast_and_keeps_channels_near_neutral() -> None:
    img = np.full((600, 600, 3), 128, dtype=np.uint8)
    img[:, :, 0] = 188
    img[:, :, 1] = 168
    img[:, :, 2] = 120

    result = enhance(img, _make_analysis(), cfg=EnhancementConfig(), strength=1.0)

    blue = float(result[:, :, 0].mean())
    green = float(result[:, :, 1].mean())
    red = float(result[:, :, 2].mean())

    assert blue < 170
    assert abs(red - blue) < 18
    assert abs(green - blue) < 20


def test_auto_lighting_lifts_dark_image() -> None:
    image = np.full((128, 128, 3), 45, dtype=np.uint8)
    analysis = {
        "white_balance": {},
        "recommendations": {
            "exposure_ev": {"value": 0.75},
            "highlights": 0.0,
            "shadows": 35.0,
        },
    }

    result = enhance(image, analysis)

    assert float(result.mean()) > float(image.mean())
    assert int(result.max()) < 255


def test_best_enhancement_controls_luminosity_and_white_balance() -> None:
    image = np.full((128, 128, 3), 40, dtype=np.uint8)
    image[:, :, 0] = 85
    image[:, :, 1] = 65
    image[:, :, 2] = 50

    analysis = {
        "white_balance": {},
        "recommendations": {"exposure_ev": {"value": 0.0}, "highlights": 0.0, "shadows": 0.0},
        "best_enhancement": {
            "luminosity": {"recommended_change_ev": 0.9, "action": "brighten", "confidence": 0.8},
            "neutral_tone": {"temperature_cast": 5.0, "tint_cast": 0.0, "confidence": 0.7},
        },
    }

    result = enhance(image, analysis)

    assert float(result.mean()) > float(image.mean())
    assert abs(float(result[:, :, 0].mean()) - float(result[:, :, 2].mean())) < 25
