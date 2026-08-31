"""
Backend configuration — BackendSettings dataclass with sensible defaults.
All paths are resolved relative to the project root (one level above backend/).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# Project root is the parent of the directory that contains this file.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BackendSettings:
    # Path to the SQLite database file
    db_path: str = os.path.join(_PROJECT_ROOT, "attendance.db")

    # Path to the face-embedding pickle store
    embeddings_file: str = os.path.join(_PROJECT_ROOT, "embeddings.pkl")

    # Root directory for collected face-image datasets
    dataset_root: str = os.path.join(_PROJECT_ROOT, "CollectedImages")

    # Directory where unknown-face JPEG crops are saved
    unknown_faces_dir: str = os.path.join(_PROJECT_ROOT, "unknown_faces")

    # CORS allowed origins (Streamlit default port)
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:8501"])

    # Uvicorn bind address
    host: str = "0.0.0.0"

    # Uvicorn bind port
    port: int = 8000


# Module-level singleton — import this wherever settings are needed.
settings = BackendSettings()
