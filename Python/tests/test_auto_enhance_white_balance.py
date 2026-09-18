import numpy as np

from autophotoeditor.processing.auto_enhance import EnhancementConfig, enhance


def _make_analysis() -> dict:
    return {
        "exposure_and_tone": {
            "median_luminance": 118.0,
            "mean_luminance": 118.0,
            "percentiles": {"p5_0": 40.0, "p95_0": 200.0},
            "highlight_clipping": {"all_channels_percentage": 0.0},
            "shadow_clipping": {"all_channels_percentage": 0.0},
            "dark_pixels_percentage": 0.0,
            "bright_pixels_percentage": 0.0,
            "effective_dynamic_range": 140.0,
        },
        "color_profile": {
            "saturation_mean": 18.0,
            "saturation_percentiles": {"p75_0": 22.0, "p90_0": 28.0},
            "high_saturation_percentage": 0.0,
            "extreme_saturation_percentage": 0.0,
            "white_balance": {
                "estimated_temperature_cast": 16.0,
                "estimated_tint_cast": 0.0,
                "neutral_pixel_coverage_percentage": 10.0,
            },
        },
        "quality": {
            "laplacian_variance": 120.0,
            "noise_estimate_std": 1.0,
        },
        "contrast": {"local_contrast": 18.0},
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
    assert red > blue
    assert abs(red - blue) < 18
    assert abs(green - blue) < 20
