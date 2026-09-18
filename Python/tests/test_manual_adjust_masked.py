import numpy as np

from autophotoeditor.processing.manual_adjust import apply_masked


def test_manual_adjustment_preserves_pixels_outside_mask() -> None:
	image = np.full((12, 12, 3), 100, dtype=np.uint8)
	mask = np.zeros((12, 12), dtype=np.float32)
	mask[3:9, 3:9] = 1.0

	result, _ = apply_masked(
		image,
		{"exposure": 1.0},
		mask,
		feather_radius=0.0,
	)

	assert np.array_equal(result[:3], image[:3])
	assert np.array_equal(result[9:], image[9:])
	assert np.mean(result[3:9, 3:9]) > np.mean(image[3:9, 3:9])
