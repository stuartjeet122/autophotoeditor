import numpy as np

from autophotoeditor.processing.extract_image_data import analyze_image


def test_neutral_gray_image_is_left_almost_unchanged() -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)

    result = analyze_image(image)

    exposure = result["exposure_and_tone"]["exposure"]["value"]
    controls = result["exposure_and_tone"]["controls"]
    plan = result["correction_plan"]

    assert abs(exposure) < 0.05
    assert abs(plan["exposure_ev"]) < 0.05
    assert all(abs(float(value)) < 2.0 for value in controls.values())
