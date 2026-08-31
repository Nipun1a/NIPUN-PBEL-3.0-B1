"""
Integration tests for the /api/settings endpoints.

Uses FastAPI's TestClient via the shared ``integration_client`` fixture
defined in conftest.py.  No sys.modules patching at module level.

Test coverage:
  - GET  /api/settings              → 200 with all 6 default keys
  - PUT  /api/settings valid update → 200, persisted value returned
  - PUT  /api/settings invalid value → 422 Unprocessable Entity

Requirements: 9.1, 9.2, 9.3
"""
from __future__ import annotations

import pytest


EXPECTED_SETTINGS_KEYS = {
    "recognition_threshold",
    "cooldown_period_seconds",
    "stable_frame_count",
    "camera_index",
    "blur_threshold",
    "min_face_size",
}


# ---------------------------------------------------------------------------
# Tests — all use `integration_client` fixture from conftest.py
# ---------------------------------------------------------------------------


def test_get_settings_returns_200_with_all_keys(integration_client):
    """GET /api/settings → 200 and response contains all 6 expected keys."""
    response = integration_client.get("/api/settings/")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body.keys()) == EXPECTED_SETTINGS_KEYS, (
        f"Missing keys: {EXPECTED_SETTINGS_KEYS - set(body.keys())}"
    )


def test_get_settings_default_recognition_threshold(integration_client):
    """GET /api/settings → recognition_threshold defaults to 0.6."""
    response = integration_client.get("/api/settings/")
    assert response.status_code == 200, response.text
    body = response.json()
    assert abs(body["recognition_threshold"] - 0.6) < 1e-9


def test_put_settings_valid_update_returns_200(integration_client):
    """PUT /api/settings with recognition_threshold=0.8 → 200 and value persisted."""
    payload = {"recognition_threshold": 0.8}
    response = integration_client.put("/api/settings/", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert abs(body["recognition_threshold"] - 0.8) < 1e-9


def test_put_settings_value_is_persisted_in_db(integration_client):
    """After PUT, GET should return the updated value."""
    integration_client.put("/api/settings/", json={"recognition_threshold": 0.75})
    response = integration_client.get("/api/settings/")
    assert response.status_code == 200, response.text
    body = response.json()
    assert abs(body["recognition_threshold"] - 0.75) < 1e-9


def test_put_settings_response_contains_all_keys(integration_client):
    """PUT /api/settings → response still contains all 6 keys."""
    response = integration_client.put("/api/settings/", json={"recognition_threshold": 0.7})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body.keys()) == EXPECTED_SETTINGS_KEYS


def test_put_settings_invalid_threshold_above_1_returns_422(integration_client):
    """PUT /api/settings with recognition_threshold=1.5 → 422."""
    payload = {"recognition_threshold": 1.5}
    response = integration_client.put("/api/settings/", json=payload)
    assert response.status_code == 422, response.text


def test_put_settings_invalid_threshold_below_0_returns_422(integration_client):
    """PUT /api/settings with recognition_threshold=-0.1 → 422."""
    payload = {"recognition_threshold": -0.1}
    response = integration_client.put("/api/settings/", json=payload)
    assert response.status_code == 422, response.text


def test_put_settings_other_keys_unchanged_after_partial_update(integration_client):
    """Updating one key should not change other keys."""
    original = integration_client.get("/api/settings/").json()
    integration_client.put("/api/settings/", json={"recognition_threshold": 0.9})
    updated = integration_client.get("/api/settings/").json()

    for key in EXPECTED_SETTINGS_KEYS - {"recognition_threshold"}:
        assert updated[key] == original[key], (
            f"Key '{key}' changed unexpectedly: {original[key]} → {updated[key]}"
        )
