"""
Database initialisation — creates all tables, indexes, and default settings rows.

`init_db()` is idempotent: it uses `CREATE TABLE IF NOT EXISTS`,
`CREATE INDEX IF NOT EXISTS`, and `INSERT OR IGNORE`, so it is safe to call
on every backend startup without duplicating schema objects or data rows.
"""
from __future__ import annotations

import aiosqlite
from datetime import datetime, timezone

from backend.config import settings

# ---------------------------------------------------------------------------
# DDL — tables
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

# ---------------------------------------------------------------------------
# DDL — indexes
# ---------------------------------------------------------------------------

_CREATE_IDX_ATTENDANCE_ROLL_DATE = """
CREATE INDEX IF NOT EXISTS idx_attendance_roll_date
    ON attendance (roll_number, date);
"""

_CREATE_IDX_ATTENDANCE_DATE = """
CREATE INDEX IF NOT EXISTS idx_attendance_date
    ON attendance (date);
"""

_CREATE_IDX_UNKNOWN_FACES_TIMESTAMP = """
CREATE INDEX IF NOT EXISTS idx_unknown_faces_timestamp
    ON unknown_faces (timestamp);
"""

# ---------------------------------------------------------------------------
# Default settings rows
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS: list[tuple[str, str]] = [
    ("recognition_threshold",   "0.6"),
    ("cooldown_period_seconds", "300"),
    ("stable_frame_count",      "4"),
    ("camera_index",            "0"),
    ("blur_threshold",          "50.0"),
    ("min_face_size",           "60"),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """
    Initialise the SQLite database at ``settings.db_path``.

    Actions performed (all idempotent):
    1. Enable WAL journal mode for better concurrent read performance.
    2. Create the four application tables if they don't already exist.
    3. Create the three query-optimisation indexes if they don't exist.
    4. Insert the six default settings rows via ``INSERT OR IGNORE`` so that
       existing customised values are never overwritten.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(settings.db_path) as db:
        # WAL mode improves read/write concurrency for the FastAPI app
        await db.execute("PRAGMA journal_mode=WAL;")

        # --- tables ---
        await db.execute(_CREATE_STUDENTS)
        await db.execute(_CREATE_ATTENDANCE)
        await db.execute(_CREATE_UNKNOWN_FACES)
        await db.execute(_CREATE_SETTINGS)

        # --- indexes ---
        await db.execute(_CREATE_IDX_ATTENDANCE_ROLL_DATE)
        await db.execute(_CREATE_IDX_ATTENDANCE_DATE)
        await db.execute(_CREATE_IDX_UNKNOWN_FACES_TIMESTAMP)

        # --- default settings (INSERT OR IGNORE — never overwrite) ---
        for key, value in _DEFAULT_SETTINGS:
            await db.execute(
                "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?);",
                (key, value, now_iso),
            )

        await db.commit()
