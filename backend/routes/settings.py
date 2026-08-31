"""
routes/settings.py

FastAPI APIRouter for the settings endpoints.

Prefix (/api/settings) is applied externally in main.py.

Endpoints:
  GET  /  → Returns the current values for all 6 settings keys as SettingsResponse
  PUT  /  → Validates via SettingsUpdate, persists changed keys, applies to the
            live Recognizer (if initialised), returns full updated SettingsResponse

Requirements: 9.1, 9.2, 9.3
"""
from __future__ import annotations

import aiosqlite
from fastapi import APIRouter, Depends

from backend.database.connection import get_db
from backend.models.settings import SettingsResponse, SettingsUpdate
from backend.services import recognition_service, settings_service

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /  → retrieve current settings
# ---------------------------------------------------------------------------


@router.get("/", response_model=SettingsResponse)
async def get_settings(
    db: aiosqlite.Connection = Depends(get_db),
) -> SettingsResponse:
    """
    Return the current value of every settings key.

    All 6 keys are always present in the response; any key that has not yet
    been written to the database is returned at its default value (merged
    inside ``settings_service.load_settings_from_db``).

    Requirements: 9.1
    """
    current = await settings_service.load_settings_from_db(db)
    return SettingsResponse(**current)


# ---------------------------------------------------------------------------
# PUT /  → update one or more settings keys
# ---------------------------------------------------------------------------


@router.put("/", response_model=SettingsResponse)
async def update_settings(
    payload: SettingsUpdate,
    db: aiosqlite.Connection = Depends(get_db),
) -> SettingsResponse:
    """
    Update one or more settings keys and return the full updated settings.

    Only the fields included in the request body (non-null) are persisted;
    omitted fields keep their current values.  Pydantic validates all
    ranges automatically and returns HTTP 422 on any out-of-range value
    before this handler is invoked.

    After persisting the changes the live Recognizer singleton is patched so
    that threshold / smoothing changes take effect immediately without a
    server restart.  If the Recognizer has not yet been initialised (e.g.
    during testing or before the lifespan startup has completed) the patch
    is silently skipped.

    Requirements: 9.2, 9.3
    """
    # Persist only the fields that were explicitly provided (exclude_none
    # drops fields whose value is None, i.e. fields not in the request body).
    updated = await settings_service.update_settings(
        payload.model_dump(exclude_none=True), db
    )

    # Attempt to apply the new settings to the live Recognizer.  Wrap in a
    # try/except so that a not-yet-initialised Recognizer does not cause the
    # PUT to fail — settings must still be persistable without the ML service.
    try:
        recognizer = recognition_service.get_recognizer()
        settings_service.apply_to_recognizer(recognizer, updated)
    except RuntimeError:
        # Recognizer not initialised yet; settings are persisted and will be
        # picked up on the next server startup via the lifespan event.
        pass

    return SettingsResponse(**updated)
