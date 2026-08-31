"""
Property-based tests for attendance service.

Property 6: Duplicate suppression is transparent
  - A second attendance insert for the same student within the cooldown
    window must be detected by is_duplicate() → True, and no new row is written.

Property 1: Attendance uniqueness within cooldown
  - Inserting multiple attendance events with varying inter-arrival times
    must result in at most one row per student per cooldown window.

Uses deterministic parametrised test cases with asyncio + in-memory aiosqlite.
"""
from __future__ import annotations

import sys
import os
import asyncio
from datetime import datetime, timedelta

import pytest
import aiosqlite

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.tests.conftest import _setup_db, _patch_db
from backend.services import attendance_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------

async def _insert_student(db, roll_number="101", name="Alice"):
    now = datetime.utcnow().isoformat()
    await db.execute(
        "INSERT INTO students (roll_number, name, department, email, phone, created_at, updated_at)"
        " VALUES (?, ?, '', '', ?, ?, ?)",
        (roll_number, name, now, now, now),
    )
    await db.commit()


async def _insert_attendance_at(db, roll_number, created_at_offset_seconds: float):
    """Insert an attendance row with a created_at offset from now."""
    created_at = (
        datetime.utcnow() - timedelta(seconds=created_at_offset_seconds)
    ).isoformat()
    today = datetime.utcnow().date().isoformat()
    await db.execute(
        """INSERT INTO attendance
               (roll_number, name, date, time, confidence_score, status, marked_by, created_at)
           VALUES (?, 'Test', ?, '09:00:00', 0.9, 'Present', 'face_recognition', ?)""",
        (roll_number, today, created_at),
    )
    await db.commit()


async def _count_today_rows(db, roll_number):
    today = datetime.utcnow().date().isoformat()
    row = await db.execute_fetchone(
        "SELECT COUNT(*) AS cnt FROM attendance WHERE roll_number = ? AND date = ?",
        (roll_number, today),
    )
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Property 6: Duplicate suppression is transparent
# ---------------------------------------------------------------------------

class TestDuplicateSuppression:
    """
    **Validates: Requirement 18.1, 18.4**
    Property 6: When two attendance events occur for the same student within
    the cooldown window, the second must be detected as a duplicate and no
    additional row must be written.
    """

    @pytest.mark.asyncio
    async def test_second_insert_detected_as_duplicate(self, db):
        """is_duplicate returns True after first row inserted within cooldown."""
        await _insert_student(db, "101")
        # Simulate first attendance recorded 30s ago (within 300s cooldown)
        await _insert_attendance_at(db, "101", created_at_offset_seconds=30)

        is_dup = await attendance_service.is_duplicate("101", 300, db)
        assert is_dup is True, "Expected True: second attempt within cooldown should be a duplicate"

    @pytest.mark.asyncio
    async def test_no_new_row_when_duplicate_detected(self, db):
        """Row count stays at 1 when the service suppresses a duplicate."""
        await _insert_student(db, "101")
        await _insert_attendance_at(db, "101", created_at_offset_seconds=30)

        count_before = await _count_today_rows(db, "101")
        assert count_before == 1

        # Service layer check: if duplicate, we do NOT call mark_attendance
        is_dup = await attendance_service.is_duplicate("101", 300, db)
        if not is_dup:
            await attendance_service.mark_attendance("101", "Alice", 0.9, db)

        count_after = await _count_today_rows(db, "101")
        assert count_after == 1, f"Expected 1 row, got {count_after} — duplicate was not suppressed"

    @pytest.mark.asyncio
    async def test_different_students_not_duplicate_of_each_other(self, db):
        """A record for student A does not make student B a duplicate."""
        await _insert_student(db, "101")
        await _insert_student(db, "102", "Bob")
        await _insert_attendance_at(db, "101", created_at_offset_seconds=30)

        is_dup_102 = await attendance_service.is_duplicate("102", 300, db)
        assert is_dup_102 is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset_seconds", [1, 60, 150, 299])
    async def test_duplicate_detected_for_various_offsets_within_cooldown(
        self, db, offset_seconds
    ):
        """Property 6 holds for any arrival within the cooldown window."""
        await _insert_student(db, "101")
        await _insert_attendance_at(db, "101", created_at_offset_seconds=offset_seconds)

        is_dup = await attendance_service.is_duplicate("101", 300, db)
        assert is_dup is True, (
            f"offset={offset_seconds}s is within 300s cooldown — should be duplicate"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset_seconds", [301, 400, 600, 3600])
    async def test_not_duplicate_when_outside_cooldown(self, db, offset_seconds):
        """After cooldown elapses the second attendance is not a duplicate."""
        await _insert_student(db, "101")
        await _insert_attendance_at(db, "101", created_at_offset_seconds=offset_seconds)

        is_dup = await attendance_service.is_duplicate("101", 300, db)
        assert is_dup is False, (
            f"offset={offset_seconds}s is outside 300s cooldown — should NOT be duplicate"
        )


# ---------------------------------------------------------------------------
# Property 1: Attendance uniqueness within cooldown
# ---------------------------------------------------------------------------

class TestAttendanceUniquenessWithinCooldown:
    """
    **Validates: Requirement 18.1, 18.2**
    Property 1: For any sequence of attendance events sent for the same student,
    the service layer (is_duplicate gate) must result in at most one row per
    student per cooldown window.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_offsets_seconds_ago, cooldown, expected_rows",
        [
            # All 3 events in the last 60s (within 300s window) → only the
            # earliest persists; the 2nd and 3rd are within the first's cooldown.
            ([60, 30, 10], 300, 1),
            # All 5 events in the last 40s (within 60s cooldown) → only first persists.
            ([50, 40, 30, 20, 10], 60, 1),
            # First event 400s ago (outside 300s cooldown), second event 10s ago → 2 rows.
            ([400, 10], 300, 2),
            # First 400s ago, second 200s ago (inside first's cooldown? no — first is
            # outside the 300s window), third 10s ago.
            # Sequence (oldest first): 400s ago → allowed; 200s ago → within 300s of now
            # but is it within 300s of 400s-ago row? The is_duplicate query checks
            # created_at > (now - cooldown), so at the moment the 2nd event arrives:
            #   now-300 = 300s ago → the 400s-ago row is NOT within window → 2nd allowed.
            #   now-300 = 300s ago → 200s-ago row IS within window → 3rd is duplicate.
            ([400, 200, 10], 300, 2),
        ],
        ids=["three_within_300s", "five_within_60s", "two_outside_and_inside", "three_mixed"],
    )
    async def test_at_most_one_row_per_cooldown_window(
        self, db, event_offsets_seconds_ago, cooldown, expected_rows
    ):
        """
        Insert attendance rows with timestamps set to specific offsets before now.
        Apply is_duplicate against those timestamps (which use the real clock) to
        determine which events would have been allowed, and verify expected row count.

        The strategy: process events from oldest to newest. Before each insert,
        call is_duplicate which checks against the *current* DB state using real
        time. Only insert if not a duplicate.
        """
        await _insert_student(db, "101")
        today = datetime.utcnow().date().isoformat()

        # Process events from oldest (largest offset) to newest (smallest offset)
        for offset in sorted(event_offsets_seconds_ago, reverse=True):
            simulated_created_at = (
                datetime.utcnow() - timedelta(seconds=offset)
            ).isoformat()
            # is_duplicate checks for any row with created_at > (now - cooldown)
            is_dup = await attendance_service.is_duplicate("101", cooldown, db)
            if not is_dup:
                await db.execute(
                    """INSERT INTO attendance
                           (roll_number, name, date, time, confidence_score,
                            status, marked_by, created_at)
                       VALUES (?, 'Alice', ?, '09:00:00', 0.9, 'Present',
                               'face_recognition', ?)""",
                    ("101", today, simulated_created_at),
                )
                await db.commit()

        total_rows = await _count_today_rows(db, "101")
        assert total_rows == expected_rows, (
            f"event_offsets={event_offsets_seconds_ago}, cooldown={cooldown}s: "
            f"expected {expected_rows} rows, got {total_rows}"
        )

    @pytest.mark.asyncio
    async def test_fresh_db_allows_first_insert(self, db):
        """The very first insert is never a duplicate."""
        await _insert_student(db, "101")
        is_dup = await attendance_service.is_duplicate("101", 300, db)
        assert is_dup is False

        await attendance_service.mark_attendance("101", "Alice", 0.9, db)
        count = await _count_today_rows(db, "101")
        assert count == 1

    @pytest.mark.asyncio
    async def test_multiple_students_independent_windows(self, db):
        """Each student's cooldown window is tracked independently."""
        await _insert_student(db, "101")
        await _insert_student(db, "102", "Bob")
        await _insert_student(db, "103", "Carol")

        # All three students arrive within the same real time — each gets their own row
        for roll in ("101", "102", "103"):
            is_dup = await attendance_service.is_duplicate(roll, 300, db)
            assert is_dup is False, f"First insert for {roll} should not be a duplicate"
            await attendance_service.mark_attendance(roll, "Student", 0.9, db)

        # Now all three should be flagged as duplicates
        for roll in ("101", "102", "103"):
            is_dup = await attendance_service.is_duplicate(roll, 300, db)
            assert is_dup is True, f"Second insert for {roll} should be a duplicate"
