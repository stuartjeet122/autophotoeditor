"""Runtime paths and application settings."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
MODEL_PATH = PROJECT_DIR / "yolo26l-seg.pt"
OUTPUT_DIR = Path(os.getenv("API_OUTPUT_DIR", PROJECT_DIR / "api_outputs"))
AI_MODEL_CACHE = Path.home() / ".cache" / "ai_denoise"
