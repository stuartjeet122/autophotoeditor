"""Command-line entry point for image analysis."""

from .engine import build_parser, main

__all__ = ["build_parser", "main"]
