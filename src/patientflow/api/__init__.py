"""REST API utilities for serving hierarchical predictions."""

from .main import app  # re-export FastAPI app for uvicorn

__all__ = ["app"]

