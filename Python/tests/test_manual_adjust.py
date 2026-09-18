import numpy as np
import pytest

from autophotoeditor.processing import manual_adjust


def test_zero_adjustments_are_identity() -> None:
	image = np.zeros((24, 32, 3), dtype=np.uint8)
	image[:, :, 0] = 40
	image[:, :, 1] = 100
	image[:, :, 2] = 220

	result, normalized = manual_adjust.apply(image, {})

	assert np.array_equal(result, image)
	assert all(value == 0.0 for value in normalized.values())


def test_adjustments_are_clamped_to_documented_ranges() -> None:
	image = np.full((24, 32, 3), 128, dtype=np.uint8)

	result, normalized = manual_adjust.apply(
		image,
		{
			"exposure": 20,
			"contrast": -200,
			"highlights": 200,
			"shadows": -200,
			"temperature": 200,
			"tint": -200,
			"saturation": 200,
			"sharpness": 200,
		},
	)

	assert result.dtype == np.uint8
	assert result.shape == image.shape
	assert normalized == {
		"exposure": 2.0,
		"contrast": -100.0,
		"highlights": 100.0,
		"shadows": -100.0,
		"temperature": 100.0,
		"tint": -100.0,
		"saturation": 100.0,
		"sharpness": 100.0,
	}


def test_manual_adjustment_changes_image_and_rejects_unknown_values() -> None:
	image = np.full((40, 40, 3), 100, dtype=np.uint8)

	result, _ = manual_adjust.apply(
		image,
		{"exposure": 1.0, "saturation": 50.0},
	)

	assert np.mean(result) > np.mean(image)

	with pytest.raises(manual_adjust.ManualAdjustmentError):
		manual_adjust.apply(image, {"unsupported": 1})
