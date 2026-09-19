"""Geometry correction processing engine."""

import sys

from . import transforms as _engine
from .canvas import (
	crop_using_mask,
	fit_to_canvas,
	render_validity_mask,
	transformed_corners,
)
from .io import read_image, write_image

_engine.read_image = read_image
_engine.write_image = write_image
_engine.crop_using_mask = crop_using_mask
_engine.fit_to_canvas = fit_to_canvas
_engine.render_validity_mask = render_validity_mask
_engine.transformed_corners = transformed_corners
sys.modules[__name__] = _engine
