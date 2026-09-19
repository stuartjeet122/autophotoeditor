"""Safe, bounded manual photographic adjustments for API clients."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


ADJUSTMENT_LIMITS: dict[str, tuple[float, float, float]] = {
	"exposure": (-2.0, 2.0, 0.0),
	"contrast": (-100.0, 100.0, 0.0),
	"highlights": (-100.0, 100.0, 0.0),
	"shadows": (-100.0, 100.0, 0.0),
	"temperature": (-100.0, 100.0, 0.0),
	"tint": (-100.0, 100.0, 0.0),
	"saturation": (-100.0, 100.0, 0.0),
	"sharpness": (0.0, 100.0, 0.0),
}


class ManualAdjustmentError(ValueError):
	"""Raised when manual adjustment input cannot be applied safely."""


def _number(value: Any, name: str) -> float:
	if isinstance(value, bool):
		raise ManualAdjustmentError(f"Adjustment '{name}' must be numeric")

	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ManualAdjustmentError(
			f"Adjustment '{name}' must be numeric"
		) from exc

	if not np.isfinite(result):
		raise ManualAdjustmentError(
			f"Adjustment '{name}' must be finite"
		)

	return result


def normalize_adjustments(
	adjustments: Mapping[str, Any],
) -> dict[str, float]:
	"""Validate and clamp the supported manual adjustment values."""
	if not isinstance(adjustments, Mapping):
		raise ManualAdjustmentError("Adjustments must be an object")

	normalized: dict[str, float] = {}

	for name, (minimum, maximum, default) in ADJUSTMENT_LIMITS.items():
		value = adjustments.get(name, default)
		numeric = _number(value, name)
		normalized[name] = float(np.clip(numeric, minimum, maximum))

	unknown = sorted(set(adjustments) - set(ADJUSTMENT_LIMITS))
	if unknown:
		raise ManualAdjustmentError(
			"Unsupported adjustment(s): " + ", ".join(unknown)
		)

	return normalized


def _apply_tonal_adjustments(
	image: np.ndarray,
	adjustments: Mapping[str, float],
) -> np.ndarray:
	"""Apply exposure, contrast, highlights, and shadows in float space."""
	result = image.astype(np.float32) / 255.0

	exposure = adjustments["exposure"]
	result *= 2.0 ** exposure

	contrast = 1.0 + adjustments["contrast"] / 100.0
	result = (result - 0.5) * contrast + 0.5

	luminance = cv2.cvtColor(
		np.clip(result, 0.0, 1.0).astype(np.float32),
		cv2.COLOR_BGR2GRAY,
	)

	highlights = adjustments["highlights"] / 100.0
	highlight_weight = np.clip((luminance - 0.45) / 0.55, 0.0, 1.0)
	result += highlights * highlight_weight[:, :, None] * 0.28

	shadows = adjustments["shadows"] / 100.0
	shadow_weight = np.clip((0.48 - luminance) / 0.48, 0.0, 1.0)
	shadow_weight = shadow_weight ** 1.45
	shadow_headroom = np.clip((0.78 - luminance) / 0.78, 0.0, 1.0)
	result += shadows * shadow_weight[:, :, None] * shadow_headroom[:, :, None] * 0.16

	return np.clip(result, 0.0, 1.0)


def _apply_color_adjustments(
	image: np.ndarray,
	adjustments: Mapping[str, float],
) -> np.ndarray:
	"""Apply temperature, tint, and saturation while preserving highlights."""
	result = image.copy()

	temperature = adjustments["temperature"] / 100.0
	tint = adjustments["tint"] / 100.0

	blue_gain = 1.0 - temperature * 0.18 + tint * 0.07
	green_gain = 1.0 - tint * 0.12
	red_gain = 1.0 + temperature * 0.18 + tint * 0.07

	result[:, :, 0] *= blue_gain
	result[:, :, 1] *= green_gain
	result[:, :, 2] *= red_gain

	hsv = cv2.cvtColor(
		np.clip(result, 0.0, 1.0).astype(np.float32),
		cv2.COLOR_BGR2HSV,
	)
	saturation_gain = 1.0 + adjustments["saturation"] / 100.0
	hsv[:, :, 1] = np.clip(
		hsv[:, :, 1] * saturation_gain,
		0.0,
		1.0,
	)

	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply(
	image: np.ndarray,
	adjustments: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
	"""Apply validated manual adjustments to an 8-bit BGR image."""
	if image is None or image.ndim != 3 or image.shape[2] != 3:
		raise ManualAdjustmentError("Expected a three-channel BGR image")

	if image.dtype != np.uint8:
		raise ManualAdjustmentError("Expected an 8-bit BGR image")

	normalized = normalize_adjustments(adjustments)
	if all(abs(value) < 1e-9 for value in normalized.values()):
		return image.copy(), normalized

	result = _apply_tonal_adjustments(image, normalized)
	result = _apply_color_adjustments(result, normalized)

	sharpness = normalized["sharpness"] / 100.0
	if sharpness > 0.0:
		blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
		result = cv2.addWeighted(
			result,
			1.0 + sharpness * 1.35,
			blurred,
			-sharpness * 1.35,
			0.0,
		)

	output = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
	return output, normalized


def apply_masked(
	image: np.ndarray,
	adjustments: Mapping[str, Any],
	mask: np.ndarray,
	feather_radius: float = 2.0,
) -> tuple[np.ndarray, dict[str, float]]:
	"""Apply manual adjustments only where a normalized mask is active."""
	if mask is None or mask.ndim != 2:
		raise ManualAdjustmentError("Expected a single-channel manual mask")

	if mask.shape != image.shape[:2]:
		raise ManualAdjustmentError("Manual mask dimensions do not match the image")

	adjusted, normalized = apply(image, adjustments)
	weight = np.clip(mask.astype(np.float32), 0.0, 1.0)

	radius = _number(feather_radius, "feather")
	if radius > 0.0:
		sigma = max(radius / 2.0, 0.5)
		weight = cv2.GaussianBlur(weight, (0, 0), sigma)
		weight = np.clip(weight, 0.0, 1.0)
		# Feather the edge without allowing the adjustment to leak into the
		# untouched part of the image.
		weight *= (mask > 0.01).astype(np.float32)

	blended = (
		image.astype(np.float32) * (1.0 - weight[:, :, None])
		+ adjusted.astype(np.float32) * weight[:, :, None]
	)

	return np.clip(blended, 0.0, 255.0).astype(np.uint8), normalized
