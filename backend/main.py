"""
FastAPI application entry point.

Responsibilities:
  - Insert project root into sys.path so the pre-existing ML modules are importable
  - Define the lifespan async context manager (DB init → load settings → init recognizer)
  - Register all 7 API routers with their prefixes and tags
  - Add CORS middleware (allow Streamlit frontend on http://localhost:8501)
  - Register global exception handlers for ValueError, KeyError, FileNotFoundError, Exception

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""
from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# IMPORTANT: Insert project root BEFORE any ML or local imports so that the
# pre-existing ML modules (recognizer, face_matcher, …) are resolvable.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiosqlite
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings as backend_settings
from backend.database.init_db import init_db
from backend.services.recognition_service import init_recognizer
from backend.services.settings_service import load_settings_from_db

# Routers
from backend.routes.students import router as students_router
from backend.routes.recognition import router as recognition_router
from backend.routes.attendance import router as attendance_router
from backend.routes.analytics import router as analytics_router
from backend.routes.export import router as export_router
from backend.routes.settings import router as settings_router
from backend.routes.unknown_faces import router as unknown_faces_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup sequence:
      1. Initialise the SQLite database (create tables + indexes + default settings).
      2. Load persisted settings from the DB.
      3. Construct the Recognizer singleton with the DB-persisted settings.

    Shutdown:
      Nothing special is required — connections are closed by their own context managers.
    """
    # 1. Create tables / indexes / default rows (idempotent)
    await init_db()

    # 2. Load settings from DB (opens its own connection)
    async with aiosqlite.connect(backend_settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        app_settings = await load_settings_from_db(db)

    # 3. Initialise the Recognizer singleton with persisted settings
    init_recognizer(app_settings)

    yield  # Application is running

    # Shutdown — no explicit teardown needed


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Attendance Monitoring System",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware — allow the Streamlit frontend (default port 8501)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(students_router,      prefix="/api/students",       tags=["Students"])
app.include_router(recognition_router,   prefix="/api/recognition",    tags=["Recognition"])
app.include_router(attendance_router,    prefix="/api/attendance",     tags=["Attendance"])
app.include_router(analytics_router,     prefix="/api/analytics",      tags=["Analytics"])
app.include_router(export_router,        prefix="/api/export",         tags=["Export"])
app.include_router(settings_router,      prefix="/api/settings",       tags=["Settings"])
app.include_router(unknown_faces_router, prefix="/api/unknown-faces",  tags=["Unknown Faces"])

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": f"Resource not found: {exc}"})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
