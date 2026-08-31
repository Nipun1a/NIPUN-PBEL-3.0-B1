"""
Unit tests for the database initialisation layer.

Tests:
  - init_db() creates all 4 expected tables
  - init_db() inserts all 6 default settings rows
  - init_db() is idempotent (safe to run twice)
  - All 3 expected indexes are created

Uses a real aiosqlite in-memory DB with the actual init_db module — but patches
the db_path to ":memory:" so we don't write to disk.
"""
import sys
import os
import asyncio
from unittest.mock import patch

import pytest
import aiosqlite

# Ensure project root on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _run_init_db_in_memory(tmp_path):
    """Run init_db() against a temp file DB (not ":memory:" — init_db opens its own conn)."""
    db_file = str(tmp_path / "test_attendance.db")

    # Patch the settings.db_path to use our temp file
    with patch("backend.database.init_db.settings") as mock_settings:
        mock_settings.db_path = db_file
        from backend.database.init_db import init_db
        await init_db()

    return db_file


class TestInitDb:
    @pytest.mark.asyncio
    async def test_all_four_tables_created(self, tmp_path):
        db_file = await _run_init_db_in_memory(tmp_path)

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cur:
                rows = await cur.fetchall()

        table_names = {row["name"] for row in rows}
        assert "students" in table_names
        assert "attendance" in table_names
        assert "unknown_faces" in table_names
        assert "settings" in table_names

    @pytest.mark.asyncio
    async def test_all_six_default_settings_rows_present(self, tmp_path):
        db_file = await _run_init_db_in_memory(tmp_path)

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT key, value FROM settings ORDER BY key") as cur:
                rows = await cur.fetchall()

        setting_keys = {row["key"] for row in rows}
        expected_keys = {
            "recognition_threshold",
            "cooldown_period_seconds",
            "stable_frame_count",
            "camera_index",
            "blur_threshold",
            "min_face_size",
        }
        assert expected_keys == setting_keys

    @pytest.mark.asyncio
    async def test_default_settings_values(self, tmp_path):
        db_file = await _run_init_db_in_memory(tmp_path)

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT key, value FROM settings") as cur:
                rows = await cur.fetchall()

        settings_dict = {row["key"]: row["value"] for row in rows}
        assert settings_dict["recognition_threshold"] == "0.6"
        assert settings_dict["cooldown_period_seconds"] == "300"
        assert settings_dict["stable_frame_count"] == "4"
        assert settings_dict["camera_index"] == "0"
        assert settings_dict["blur_threshold"] == "50.0"
        assert settings_dict["min_face_size"] == "60"

    @pytest.mark.asyncio
    async def test_idempotent_run_twice(self, tmp_path):
        """Running init_db() twice must not raise and must not duplicate settings rows."""
        db_file = await _run_init_db_in_memory(tmp_path)

        # Run again against the same file
        with patch("backend.database.init_db.settings") as mock_settings:
            mock_settings.db_path = db_file
            from backend.database.init_db import init_db
            await init_db()  # second call — should be a no-op

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT COUNT(*) AS cnt FROM settings") as cur:
                row = await cur.fetchone()

        # Should still be exactly 6 rows — INSERT OR IGNORE prevents duplicates
        assert row["cnt"] == 6

    @pytest.mark.asyncio
    async def test_idempotent_preserves_custom_settings(self, tmp_path):
        """Running init_db() a second time must not overwrite already-customised settings."""
        db_file = await _run_init_db_in_memory(tmp_path)

        # Manually update a setting to a non-default value
        async with aiosqlite.connect(db_file) as db:
            await db.execute(
                "UPDATE settings SET value = '0.9' WHERE key = 'recognition_threshold'"
            )
            await db.commit()

        # Run init_db again
        with patch("backend.database.init_db.settings") as mock_settings:
            mock_settings.db_path = db_file
            from backend.database.init_db import init_db
            await init_db()

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT value FROM settings WHERE key = 'recognition_threshold'"
            ) as cur:
                row = await cur.fetchone()

        assert row["value"] == "0.9"  # custom value preserved

    @pytest.mark.asyncio
    async def test_all_three_indexes_created(self, tmp_path):
        db_file = await _run_init_db_in_memory(tmp_path)

        async with aiosqlite.connect(db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ) as cur:
                rows = await cur.fetchall()

        index_names = {row["name"] for row in rows}
        assert "idx_attendance_roll_date" in index_names
        assert "idx_attendance_date" in index_names
        assert "idx_unknown_faces_timestamp" in index_names
