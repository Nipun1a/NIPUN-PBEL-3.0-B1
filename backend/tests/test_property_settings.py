"""
Property-based test for settings bounds enforcement.

Property 4: Settings bounds enforcement
  For each of the 6 settings keys, out-of-range values MUST return HTTP 422
  and in-range values MUST return HTTP 200. On a 422 response the database
  value for that key MUST remain unchanged (no mutation on validation error).

Validates: Requirements 9.3, 9.4, 9.5, 9.6

Uses pytest + FastAPI TestClient with an in-memory SQLite database injected
via FastAPI dependency overrides. The recognition_service._recognizer is left
as None so get_recognizer() raises RuntimeError; the PUT handler catches this
gracefully and still returns 200.
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path before any backend imports.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
from typing import Any
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings as hyp_settings, HealthCheck
from hypothesis import strategies as st

from backend.main import app
from backend.database.connection import get_db
from backend.tests.conftest import _setup_db, _patch_db


# ---------------------------------------------------------------------------
# Helpers — in-memory DB fixture wired to TestClient
# ---------------------------------------------------------------------------

def make_test_client() -> tuple[TestClient, aiosqlite.Connection]:
    """
    Build a synchronous in-memory DB connection and override get_db with it,
    then return a TestClient.  The connection is created synchronously via the
    event loop so it can be shared across multiple HTTP calls within one test.
    """
    loop = asyncio.new_event_loop()

    async def _open_db():
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await _setup_db(conn)
        _patch_db(conn)
        return conn

    conn = loop.run_until_complete(_open_db())

    async def override_get_db():
        yield conn

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app, raise_server_exceptions=False)
    return client, conn, loop


def close_test_client(conn, loop):
    async def _close():
        await conn.close()

    loop.run_until_complete(_close())
    loop.close()
    app.dependency_overrides.clear()


def read_db_value(conn, loop, key: str) -> str | None:
    """Return the raw string value of a settings key from the DB."""
    async def _fetch():
        async with conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    return loop.run_until_complete(_fetch())


# ---------------------------------------------------------------------------
# Settings bounds specification
# ---------------------------------------------------------------------------
# Each entry: (key, low_invalid, high_invalid, valid_value, low_inclusive, high_inclusive)
# low_inclusive / high_inclusive = whether the boundary itself is valid (True) or invalid (False)

SETTINGS_SPEC: list[dict] = [
    {
        "key": "recognition_threshold",
        "field_type": "float",
        "ge": 0.0,
        "le": 1.0,
        "below_boundary": -0.01,   # invalid  (< 0.0)
        "above_boundary": 1.01,    # invalid  (> 1.0)
        "valid_value": 0.5,
    },
    {
        "key": "cooldown_period_seconds",
        "field_type": "int",
        "ge": 0,
        "le": 86400,
        "below_boundary": -1,      # invalid  (< 0)
        "above_boundary": 86401,   # invalid  (> 86400)
        "valid_value": 300,
    },
    {
        "key": "stable_frame_count",
        "field_type": "int",
        "ge": 1,
        "le": 30,
        "below_boundary": 0,       # invalid  (< 1)
        "above_boundary": 31,      # invalid  (> 30)
        "valid_value": 5,
    },
    {
        "key": "camera_index",
        "field_type": "int",
        "ge": 0,
        "le": 9,
        "below_boundary": -1,      # invalid  (< 0)
        "above_boundary": 10,      # invalid  (> 9)
        "valid_value": 2,
    },
    {
        "key": "blur_threshold",
        "field_type": "float",
        "ge": 0.0,
        "le": 500.0,
        "below_boundary": -1.0,    # invalid  (< 0.0)
        "above_boundary": 501.0,   # invalid  (> 500.0)
        "valid_value": 50.0,
    },
    {
        "key": "min_face_size",
        "field_type": "int",
        "ge": 10,
        "le": 500,
        "below_boundary": 9,       # invalid  (< 10)
        "above_boundary": 501,     # invalid  (> 500)
        "valid_value": 60,
    },
]


# ---------------------------------------------------------------------------
# Parametrised boundary tests (deterministic, fast)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_below_boundary_returns_422(spec):
    """A value strictly below the ge bound must return HTTP 422."""
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: spec["below_boundary"]},
            )
        assert resp.status_code == 422, (
            f"{spec['key']}={spec['below_boundary']} expected 422, got {resp.status_code}: {resp.text}"
        )
    finally:
        close_test_client(conn, loop)


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_above_boundary_returns_422(spec):
    """A value strictly above the le bound must return HTTP 422."""
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: spec["above_boundary"]},
            )
        assert resp.status_code == 422, (
            f"{spec['key']}={spec['above_boundary']} expected 422, got {resp.status_code}: {resp.text}"
        )
    finally:
        close_test_client(conn, loop)


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_valid_value_returns_200(spec):
    """An in-range value must return HTTP 200."""
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: spec["valid_value"]},
            )
        assert resp.status_code == 200, (
            f"{spec['key']}={spec['valid_value']} expected 200, got {resp.status_code}: {resp.text}"
        )
    finally:
        close_test_client(conn, loop)


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_no_db_mutation_on_422_below(spec):
    """DB value must not change after a 422 caused by a below-boundary input."""
    client, conn, loop = make_test_client()
    try:
        # Record initial DB value
        initial_value = read_db_value(conn, loop, spec["key"])

        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: spec["below_boundary"]},
            )
        assert resp.status_code == 422

        # DB value must be unchanged
        after_value = read_db_value(conn, loop, spec["key"])
        assert after_value == initial_value, (
            f"{spec['key']}: DB value changed after 422. "
            f"Before={initial_value!r}, After={after_value!r}"
        )
    finally:
        close_test_client(conn, loop)


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_no_db_mutation_on_422_above(spec):
    """DB value must not change after a 422 caused by an above-boundary input."""
    client, conn, loop = make_test_client()
    try:
        initial_value = read_db_value(conn, loop, spec["key"])

        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: spec["above_boundary"]},
            )
        assert resp.status_code == 422

        after_value = read_db_value(conn, loop, spec["key"])
        assert after_value == initial_value, (
            f"{spec['key']}: DB value changed after 422. "
            f"Before={initial_value!r}, After={after_value!r}"
        )
    finally:
        close_test_client(conn, loop)


# ---------------------------------------------------------------------------
# Property-based tests using Hypothesis
# ---------------------------------------------------------------------------
# **Validates: Requirements 9.3, 9.4, 9.5, 9.6**


def _float_out_of_range(ge: float, le: float):
    """Strategy: floats strictly outside [ge, le]."""
    return st.one_of(
        st.floats(max_value=ge - 0.001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=le + 0.001, allow_nan=False, allow_infinity=False),
    )


def _int_out_of_range(ge: int, le: int):
    """Strategy: integers strictly outside [ge, le]."""
    return st.one_of(
        st.integers(max_value=ge - 1),
        st.integers(min_value=le + 1),
    )


def _float_in_range(ge: float, le: float):
    return st.floats(min_value=ge, max_value=le, allow_nan=False, allow_infinity=False)


def _int_in_range(ge: int, le: int):
    return st.integers(min_value=ge, max_value=le)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_float_out_of_range(0.0, 1.0))
def test_property_recognition_threshold_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.4**
    Any recognition_threshold outside [0.0, 1.0] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"recognition_threshold": value})
        assert resp.status_code == 422, (
            f"recognition_threshold={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_float_in_range(0.0, 1.0))
def test_property_recognition_threshold_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.4**
    Any recognition_threshold inside [0.0, 1.0] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"recognition_threshold": value})
        assert resp.status_code == 200, (
            f"recognition_threshold={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_out_of_range(0, 86400))
def test_property_cooldown_period_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.5**
    Any cooldown_period_seconds outside [0, 86400] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"cooldown_period_seconds": value})
        assert resp.status_code == 422, (
            f"cooldown_period_seconds={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_in_range(0, 86400))
def test_property_cooldown_period_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.5**
    Any cooldown_period_seconds inside [0, 86400] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"cooldown_period_seconds": value})
        assert resp.status_code == 200, (
            f"cooldown_period_seconds={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_out_of_range(1, 30))
def test_property_stable_frame_count_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any stable_frame_count outside [1, 30] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"stable_frame_count": value})
        assert resp.status_code == 422, (
            f"stable_frame_count={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_in_range(1, 30))
def test_property_stable_frame_count_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any stable_frame_count inside [1, 30] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"stable_frame_count": value})
        assert resp.status_code == 200, (
            f"stable_frame_count={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_out_of_range(0, 9))
def test_property_camera_index_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any camera_index outside [0, 9] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"camera_index": value})
        assert resp.status_code == 422, (
            f"camera_index={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_in_range(0, 9))
def test_property_camera_index_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any camera_index inside [0, 9] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"camera_index": value})
        assert resp.status_code == 200, (
            f"camera_index={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_float_out_of_range(0.0, 500.0))
def test_property_blur_threshold_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any blur_threshold outside [0.0, 500.0] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"blur_threshold": value})
        assert resp.status_code == 422, (
            f"blur_threshold={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_float_in_range(0.0, 500.0))
def test_property_blur_threshold_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any blur_threshold inside [0.0, 500.0] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"blur_threshold": value})
        assert resp.status_code == 200, (
            f"blur_threshold={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_out_of_range(10, 500))
def test_property_min_face_size_out_of_range_gives_422(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any min_face_size outside [10, 500] must return 422.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"min_face_size": value})
        assert resp.status_code == 422, (
            f"min_face_size={value} expected 422, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


@hyp_settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(value=_int_in_range(10, 500))
def test_property_min_face_size_in_range_gives_200(value):
    """
    **Validates: Requirements 9.3, 9.6**
    Any min_face_size inside [10, 500] must return 200.
    """
    client, conn, loop = make_test_client()
    try:
        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put("/api/settings/", json={"min_face_size": value})
        assert resp.status_code == 200, (
            f"min_face_size={value} expected 200, got {resp.status_code}"
        )
    finally:
        close_test_client(conn, loop)


# ---------------------------------------------------------------------------
# Task 9.7: Settings persistence tests
# ---------------------------------------------------------------------------
# These two tests verify:
#   1. PUT settings value → DB stores the new value
#   2. GET after PUT → API returns the updated value


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_put_persists_value_in_db(spec):
    """
    **Validates: Requirement 9.1, 9.2**
    After a successful PUT the database must store the new value for that key.
    """
    client, conn, loop = make_test_client()
    try:
        new_value = spec["valid_value"]

        with patch("backend.services.recognition_service._recognizer", None):
            resp = client.put(
                "/api/settings/",
                json={spec["key"]: new_value},
            )
        assert resp.status_code == 200, (
            f"PUT {spec['key']}={new_value} expected 200, got {resp.status_code}: {resp.text}"
        )

        # Read the DB directly and verify the value was persisted
        db_value = read_db_value(conn, loop, spec["key"])
        assert db_value == str(new_value), (
            f"{spec['key']}: DB has {db_value!r}, expected {str(new_value)!r}"
        )
    finally:
        close_test_client(conn, loop)


@pytest.mark.parametrize("spec", SETTINGS_SPEC, ids=[s["key"] for s in SETTINGS_SPEC])
def test_get_after_put_returns_updated_value(spec):
    """
    **Validates: Requirement 9.1, 9.2**
    A GET /api/settings/ after a valid PUT must return the updated value.
    """
    client, conn, loop = make_test_client()
    try:
        new_value = spec["valid_value"]

        with patch("backend.services.recognition_service._recognizer", None):
            put_resp = client.put(
                "/api/settings/",
                json={spec["key"]: new_value},
            )
        assert put_resp.status_code == 200

        with patch("backend.services.recognition_service._recognizer", None):
            get_resp = client.get("/api/settings/")
        assert get_resp.status_code == 200

        returned_settings = get_resp.json()
        # The GET response may return numeric or string; compare as string for robustness
        returned_val = str(returned_settings.get(spec["key"], ""))
        expected_val = str(new_value)
        assert returned_val == expected_val, (
            f"GET after PUT: {spec['key']} expected {expected_val!r}, got {returned_val!r}"
        )
    finally:
        close_test_client(conn, loop)
