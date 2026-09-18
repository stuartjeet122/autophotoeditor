"""Models and validation for the multi-stage processing pipeline."""

from __future__ import annotations

from enum import StrEnum


class PipelineStage(StrEnum):
    ANALYZE = "analyze"
    AUTO_ENHANCE = "auto-enhance"
    DENOISE = "denoise"
    GEOMETRY = "geometry"

    @classmethod
    def parse_many(cls, stages: str) -> list["PipelineStage"]:
        parsed: list[PipelineStage] = []
        for value in stages.split(","):
            value = value.strip().lower()
            if value:
                try:
                    parsed.append(cls(value))
                except ValueError as exc:
                    raise ValueError(f"Unsupported pipeline stage: {value}") from exc
        return parsed
