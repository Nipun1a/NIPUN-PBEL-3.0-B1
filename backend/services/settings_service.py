"""
settings_service.py — Settings persistence and live Recognizer update.

Provides:
    load_settings_from_db(db)       — reads settings table, merges with defaults
    update_settings(updates, db)    — persists provided keys, returns full settings dict
    apply_to_recognizer(recognizer, settings) — patches live Recognizer attributes
"""

from datetime import datetime, timezone
from typing import Any

# Default values for all 6 settings keys.
# These are used when a key is missing from the DB (e.g. fresh install before init_db).
DEFAULTS: dict[str, Any] = {
    "recognition_threshold": 0.6,
    "cooldown_period_seconds": 300,
    "stable_frame_count": 4,
    "camera_index": 0,
    "blur_threshold": 50.0,
    "min_face_size": 60,
}

# Keys that should be cast to float; all others are cast to int.
_FLOAT_KEYS = {"recognition_threshold", "blur_threshold"}


def _cast(key: str, value: str) -> Any:
    """Cast a raw string value from the DB to the correct Python type."""
    if key in _FLOAT_KEYS:
        return float(value)
    return int(value)


async def load_settings_from_db(db) -> dict:
    """
    Read all rows from the ``settings`` table and return a fully-populated
    settings dict with correct Python types.

    Any key that is absent from the DB is filled in from DEFAULTS so the
    caller always receives all 6 keys regardless of DB state.

    Args:
        db: An open ``aiosqlite.Connection``.

    Returns:
        dict with keys: recognition_threshold (float), cooldown_period_seconds (int),
        stable_frame_count (int), camera_index (int), blur_threshold (float),
        min_face_size (int).
    """
    cursor = await db.execute("SELECT key, value FROM settings")
    rows = await cursor.fetchall()

    # Build a dict from DB rows, casting to the correct type.
    db_settings: dict[str, Any] = {}
    for row in rows:
        key = row[0]
        raw_value = row[1]
        if key in DEFAULTS:
            db_settings[key] = _cast(key, raw_value)

    # Merge: start from defaults, then overlay whatever the DB has.
    merged = dict(DEFAULTS)
    merged.update(db_settings)

    return merged


async def update_settings(updates: dict, db) -> dict:
    """
    Persist only the provided (non-None) keys to the ``settings`` table and
    return the full updated settings dict.

    Uses ``INSERT OR REPLACE`` so the upsert is safe to call repeatedly.

    Args:
        updates: A dict of settings keys to new values.  Keys with a ``None``
                 value are silently skipped.
        db:      An open ``aiosqlite.Connection``.

    Returns:
        Full settings dict (all 6 keys) after the update has been applied.
    """
    now = datetime.now(timezone.utc).isoformat()

    for key, value in updates.items():
        if value is None:
            continue
        if key not in DEFAULTS:
            # Ignore unknown keys to avoid polluting the settings table.
            continue
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), now),
        )

    await db.commit()

    # Re-read the full settings so the return value reflects what is now in DB.
    return await load_settings_from_db(db)


def apply_to_recognizer(recognizer, settings: dict) -> None:
    """
    Patch the live ``Recognizer`` singleton with the supplied settings dict
    so that changes take effect immediately without a server restart.

    Attributes updated:
        * ``recognizer._recognition_threshold``
        * ``recognizer._matcher._threshold``  (if the recognizer has a ``_matcher``)
        * ``recognizer._smoothing_window``     (if the recognizer has the attribute)

    Args:
        recognizer: The live ``Recognizer`` instance managed by
                    ``recognition_service``.
        settings:   Full settings dict as returned by ``load_settings_from_db``
                    or ``update_settings``.
    """
    recognition_threshold = float(settings["recognition_threshold"])
    stable_frame_count = int(settings["stable_frame_count"])

    # Always patch the top-level threshold on the recognizer.
    recognizer._recognition_threshold = recognition_threshold

    # Patch the inner matcher threshold if the recognizer exposes one.
    if hasattr(recognizer, "_matcher"):
        recognizer._matcher._threshold = recognition_threshold

    # Patch the smoothing window if the attribute exists.
    # Note: this affects new smoothing buffers; existing per-track buffers
    # retain their current size until they are recycled.
    if hasattr(recognizer, "_smoothing_window"):
        recognizer._smoothing_window = stable_frame_count
