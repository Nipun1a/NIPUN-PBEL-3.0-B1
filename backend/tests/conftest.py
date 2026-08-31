"""
Shared pytest fixtures for backend service and model tests.

Uses an in-memory SQLite database (aiosqlite) so tests are fast and isolated.
Each test gets a fresh database via the ``db`` fixture.
"""
import sys
import os

# Ensure the project root is on sys.path so backend package is importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import pytest
import aiosqlite


# ---------------------------------------------------------------------------
# In-memory DB fixture — run DDL from init_db but against ":memory:"
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


async def _setup_db(db: aiosqlite.Connection) -> None:
    """Create all tables and insert default settings rows."""
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


# ---------------------------------------------------------------------------
# Helpers added directly to the connection to mirror init_db's db interface
# ---------------------------------------------------------------------------

async def _execute_fetchone(db: aiosqlite.Connection, sql: str, params=()):
    """Convenience: execute query and return first row (or None)."""
    async with db.execute(sql, params) as cur:
        row = await cur.fetchone()
    return row


async def _execute_fetchall(db: aiosqlite.Connection, sql: str, params=()):
    """Convenience: execute query and return all rows."""
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return rows


def _patch_db(db: aiosqlite.Connection):
    """Add helper methods that the service layer calls on the db object."""
    import types

    db.execute_fetchone = types.MethodType(
        lambda self, sql, params=(): _execute_fetchone(self, sql, params), db
    )
    db.execute_fetchall = types.MethodType(
        lambda self, sql, params=(): _execute_fetchall(self, sql, params), db
    )
    return db


@pytest.fixture
def event_loop():
    """Create a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db(event_loop):
    """
    Provide a fresh in-memory aiosqlite connection with all tables created.
    Each test gets its own isolated database.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _setup_db(conn)
        _patch_db(conn)
        yield conn
