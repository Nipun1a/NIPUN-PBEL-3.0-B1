"""
Unit tests for backend/services/settings_service.py.

Tests:
  - load_settings_from_db: returns all 6 keys with correct types
  - load_settings_from_db: fills in defaults for missing keys
  - update_settings: persists provided keys to DB
  - update_settings: returns full settings dict after update
  - update_settings: ignores None values
  - update_settings: ignores unknown keys
  - apply_to_recognizer: patches threshold and smoothing window attributes
"""
import sys
import os
import asyncio
from unittest.mock import MagicMock

import pytest
import aiosqlite

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.tests.conftest import _setup_db, _patch_db
from backend.services import settings_service
from backend.services.settings_service import DEFAULTS


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db():
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _setup_db(conn)
        _patch_db(conn)
        yield conn


# ---------------------------------------------------------------------------
# load_settings_from_db
# ---------------------------------------------------------------------------

class TestLoadSettingsFromDb:
    @pytest.mark.asyncio
    async def test_returns_all_six_keys(self, db):
        result = await settings_service.load_settings_from_db(db)
        assert set(result.keys()) == set(DEFAULTS.keys())

    @pytest.mark.asyncio
    async def test_default_values_correct(self, db):
        result = await settings_service.load_settings_from_db(db)
        assert result["recognition_threshold"] == pytest.approx(0.6)
        assert result["cooldown_period_seconds"] == 300
        assert result["stable_frame_count"] == 4
        assert result["camera_index"] == 0
        assert result["blur_threshold"] == pytest.approx(50.0)
        assert result["min_face_size"] == 60

    @pytest.mark.asyncio
    async def test_float_keys_are_floats(self, db):
        result = await settings_service.load_settings_from_db(db)
        assert isinstance(result["recognition_threshold"], float)
        assert isinstance(result["blur_threshold"], float)

    @pytest.mark.asyncio
    async def test_int_keys_are_ints(self, db):
        result = await settings_service.load_settings_from_db(db)
        assert isinstance(result["cooldown_period_seconds"], int)
        assert isinstance(result["stable_frame_count"], int)
        assert isinstance(result["camera_index"], int)
        assert isinstance(result["min_face_size"], int)

    @pytest.mark.asyncio
    async def test_fills_missing_key_with_default(self, db):
        # Remove one key from DB
        await db.execute("DELETE FROM settings WHERE key = 'min_face_size'")
        await db.commit()

        result = await settings_service.load_settings_from_db(db)
        assert result["min_face_size"] == DEFAULTS["min_face_size"]

    @pytest.mark.asyncio
    async def test_custom_value_overrides_default(self, db):
        await db.execute(
            "UPDATE settings SET value = '0.75' WHERE key = 'recognition_threshold'"
        )
        await db.commit()

        result = await settings_service.load_settings_from_db(db)
        assert result["recognition_threshold"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# update_settings
# ---------------------------------------------------------------------------

class TestUpdateSettings:
    @pytest.mark.asyncio
    async def test_updates_single_key(self, db):
        result = await settings_service.update_settings(
            {"recognition_threshold": 0.8}, db
        )
        assert result["recognition_threshold"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_persists_value_to_db(self, db):
        await settings_service.update_settings({"cooldown_period_seconds": 600}, db)

        async with db.execute(
            "SELECT value FROM settings WHERE key = 'cooldown_period_seconds'"
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == "600"

    @pytest.mark.asyncio
    async def test_returns_full_settings_dict(self, db):
        result = await settings_service.update_settings(
            {"stable_frame_count": 8}, db
        )
        assert set(result.keys()) == set(DEFAULTS.keys())

    @pytest.mark.asyncio
    async def test_none_values_ignored(self, db):
        # None should not overwrite existing value
        original = await settings_service.load_settings_from_db(db)
        await settings_service.update_settings({"recognition_threshold": None}, db)
        after = await settings_service.load_settings_from_db(db)
        assert after["recognition_threshold"] == original["recognition_threshold"]

    @pytest.mark.asyncio
    async def test_unknown_keys_ignored(self, db):
        # Should not raise, should not insert a new row
        await settings_service.update_settings({"nonexistent_key": 42}, db)

        async with db.execute("SELECT COUNT(*) AS cnt FROM settings") as cur:
            row = await cur.fetchone()
        assert row[0] == 6  # still exactly 6 rows

    @pytest.mark.asyncio
    async def test_multiple_keys_updated_together(self, db):
        result = await settings_service.update_settings(
            {"recognition_threshold": 0.7, "cooldown_period_seconds": 600}, db
        )
        assert result["recognition_threshold"] == pytest.approx(0.7)
        assert result["cooldown_period_seconds"] == 600


# ---------------------------------------------------------------------------
# apply_to_recognizer
# ---------------------------------------------------------------------------

class TestApplyToRecognizer:
    def test_patches_recognition_threshold(self):
        recognizer = MagicMock()
        settings_service.apply_to_recognizer(
            recognizer, {"recognition_threshold": 0.75, "stable_frame_count": 4}
        )
        assert recognizer._recognition_threshold == pytest.approx(0.75)

    def test_patches_matcher_threshold_if_present(self):
        recognizer = MagicMock()
        matcher = MagicMock()
        recognizer._matcher = matcher

        settings_service.apply_to_recognizer(
            recognizer, {"recognition_threshold": 0.8, "stable_frame_count": 4}
        )
        assert matcher._threshold == pytest.approx(0.8)

    def test_patches_smoothing_window_if_present(self):
        recognizer = MagicMock()
        recognizer._smoothing_window = 4

        settings_service.apply_to_recognizer(
            recognizer, {"recognition_threshold": 0.6, "stable_frame_count": 10}
        )
        assert recognizer._smoothing_window == 10

    def test_no_error_when_matcher_attribute_missing(self):
        """Recognizer without _matcher should not raise."""
        class SimpleRecognizer:
            _recognition_threshold = 0.6

        rec = SimpleRecognizer()
        # Should not raise
        settings_service.apply_to_recognizer(
            rec, {"recognition_threshold": 0.7, "stable_frame_count": 5}
        )
        assert rec._recognition_threshold == pytest.approx(0.7)
