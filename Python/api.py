"""Compatibility entrypoint for running AutoPhotoEditor with Uvicorn."""

from autophotoeditor.api.app import app

__all__ = ["app"]
