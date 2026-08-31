"""
conftest.py for integration tests.

Provides shared fixtures for all integration tests WITHOUT patching sys.modules
at module level (which permanently corrupts the module registry for unit tests
running in the same process).

Instead, all ML-related patching is done via unittest.mock.patch() inside the
``integration_client`` fixture, which is scoped per-test and cleaned up afterward.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Shared DB DDL used by all integration test files
# ---------------------------------------------------------------------------
_CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    roll_number TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT DEFAULT '',
    email       TEXT DEFAULT '',
    phone       TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""
_CREATE_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number      TEXT    NOT NULL REFERENCES students(roll_number),
    name             TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    time             TEXT    NOT NULL,
    confidence_score REAL    DEFAULT 0.0,
    status           TEXT    NOT NULL DEFAULT 'Present',
    marked_by        TEXT    NOT NULL DEFAULT 'face_recognition',
    created_at       TEXT    NOT NULL
);
"""
_CREATE_UNKNOWN_FACES = """
CREATE TABLE IF NOT EXISTS unknown_faces (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    confidence_score REAL    DEFAULT 0.0,
    image_path       TEXT    DEFAULT '',
    created_at       TEXT    NOT NULL
);
"""
_CREATE_SETTINGS = """
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
_DEFAULT_SETTINGS = [
    ("recognition_threshold",   "0.6"),
    ("cooldown_period_seconds", "300"),
    ("stable_frame_count",      "4"),
    ("camera_index",            "0"),
    ("blur_threshold",          "50.0"),
    ("min_face_size",           "60"),
]


async def make_in_memory_db() -> aiosqlite.Connection:
    """Create a fully-initialised in-memory aiosqlite connection."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute(_CREATE_STUDENTS)
    await db.execute(_CREATE_ATTENDANCE)
    await db.execute(_CREATE_UNKNOWN_FACES)
    await db.execute(_CREATE_SETTINGS)
    for key, value in _DEFAULT_SETTINGS:
        await db.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, "2024-01-01T00:00:00+00:00"),
        )
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_db():
    """Provide a fresh in-memory SQLite DB for one test."""
    loop = asyncio.new_event_loop()
    db = loop.run_until_complete(make_in_memory_db())
    yield db, loop
    loop.run_until_complete(db.close())
    loop.close()


@pytest.fixture
def integration_client(integration_db):
    """
    Provide a FastAPI TestClient with:
    - An isolated in-memory SQLite DB via dependency override
    - Lifespan (init_db, init_recognizer) patched to no-ops
    - recognition_service._recognizer replaced with a MagicMock via patch()

    All patches are applied via context managers so they are reversed after
    each test — no permanent sys.modules corruption.
    """
    # Import here (not at module level) so unit tests that load this conftest
    # without needing the app do not trigger heavy imports.
    from backend.main import app
    from backend.database.connection import get_db

    db_conn, _loop = integration_db

    async def _override_get_db():
        yield db_conn

    app.dependency_overrides[get_db] = _override_get_db

    mock_recognizer = MagicMock()
    mock_recognizer.reload_embeddings = MagicMock()
    mock_recognizer._store = {}

    from fastapi.testclient import TestClient

    with patch("backend.main.init_db", return_value=None), \
         patch("backend.main.init_recognizer", return_value=None), \
         patch("backend.main.load_settings_from_db", return_value={}), \
         patch("backend.services.recognition_service._recognizer", mock_recognizer):
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def client_and_db(integration_db):
    """
    Legacy fixture name used by test_api_unknown_faces.py.
    Yields (TestClient, db_conn, loop).
    """
    from backend.main import app
    from backend.database.connection import get_db
    from fastapi.testclient import TestClient
    from contextlib import asynccontextmanager

    db_conn, loop = integration_db

    async def _override_get_db():
        yield db_conn

    @asynccontextmanager
    async def _noop_lifespan(application):
        yield

    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_db] = _override_get_db

    mock_recognizer = MagicMock()
    mock_recognizer.reload_embeddings = MagicMock()
    mock_recognizer._store = {}

    with patch("backend.services.recognition_service._recognizer", mock_recognizer):
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, db_conn, loop

    app.dependency_overrides.clear()
