import numpy as np

from autophotoeditor.processing.extract_image_data import analyze_image


def test_neutral_gray_image_is_left_almost_unchanged() -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)

    result = analyze_image(image)

    exposure = result["exposure_and_tone"]["exposure"]["value"]
    controls = result["exposure_and_tone"]["controls"]
    plan = result["correction_plan"]

    assert abs(exposure["value"]) < 0.05
    assert abs(plan["exposure_ev"]) < 0.05
    assert all(abs(float(value)) < 2.0 for value in controls.values())


def test_contrasty_scene_keeps_tone_controls_conservative() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:] = 40
    image[:, 150:] = 200
    image[60:140, 60:140] = 15
    image[30:60, 150:190] = 240
    image[:, 180:] = 255

    result = analyze_image(image)

    exposure = result["exposure_and_tone"]["exposure"]
    controls = result["exposure_and_tone"]["controls"]

    assert abs(float(exposure["value"])) < 0.25
    assert abs(float(controls["shadows"])) < 0.6
    assert abs(float(controls["blacks"])) < 0.8
    assert abs(float(controls["highlights"])) < 1.0
