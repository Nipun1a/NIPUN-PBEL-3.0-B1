"""
Unit tests for backend/services/analytics_service.py.

Tests:
  - get_dashboard_stats: correct counts on empty DB
  - get_dashboard_stats: correct counts with seeded data
  - get_dashboard_stats: attendance_percentage calculation
  - get_trends: returns daily trends
  - get_trends: returns weekly and monthly trends without error
  - get_department_stats: groups by department
  - get_student_stats: raises KeyError for unknown student
  - get_student_stats: returns correct present/absent counts
  - get_heatmap: returns 90 days, marks present/absent correctly
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone, date

import pytest
import aiosqlite

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.tests.conftest import _setup_db, _patch_db
from backend.services import analytics_service


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


async def _add_student(db, roll_number, name, department="CS", created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO students (roll_number, name, department, email, phone, created_at, updated_at)"
        " VALUES (?, ?, ?, '', '', ?, ?)",
        (roll_number, name, department, created_at, created_at),
    )
    await db.commit()


async def _add_attendance(db, roll_number, name, date_str=None, time_str="09:00:00", created_at=None):
    if date_str is None:
        date_str = datetime.utcnow().date().isoformat()
    if created_at is None:
        created_at = datetime.utcnow().isoformat()
    await db.execute(
        "INSERT INTO attendance (roll_number, name, date, time, confidence_score, status, marked_by, created_at)"
        " VALUES (?, ?, ?, ?, 0.9, 'Present', 'face_recognition', ?)",
        (roll_number, name, date_str, time_str, created_at),
    )
    await db.commit()


async def _add_unknown_face(db, timestamp=None):
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    await db.execute(
        "INSERT INTO unknown_faces (timestamp, confidence_score, image_path, created_at)"
        " VALUES (?, 0.3, '', ?)",
        (timestamp, timestamp),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# get_dashboard_stats
# ---------------------------------------------------------------------------

class TestGetDashboardStats:
    @pytest.mark.asyncio
    async def test_empty_db_returns_zeros(self, db):
        result = await analytics_service.get_dashboard_stats(db)
        assert result["total_students"] == 0
        assert result["present_today"] == 0
        assert result["absent_today"] == 0
        assert result["attendance_percentage"] == 0.0
        assert result["unknown_face_count"] == 0

    @pytest.mark.asyncio
    async def test_counts_students_correctly(self, db):
        await _add_student(db, "101", "Alice")
        await _add_student(db, "102", "Bob")

        result = await analytics_service.get_dashboard_stats(db)
        assert result["total_students"] == 2

    @pytest.mark.asyncio
    async def test_present_today_counts_distinct_roll_numbers(self, db):
        await _add_student(db, "101", "Alice")
        # Two records for same student (same day) — should count as 1
        await _add_attendance(db, "101", "Alice")
        await _add_attendance(db, "101", "Alice", time_str="10:00:00")

        result = await analytics_service.get_dashboard_stats(db)
        assert result["present_today"] == 1

    @pytest.mark.asyncio
    async def test_absent_today_is_total_minus_present(self, db):
        await _add_student(db, "101", "Alice")
        await _add_student(db, "102", "Bob")
        await _add_attendance(db, "101", "Alice")  # only Alice present

        result = await analytics_service.get_dashboard_stats(db)
        assert result["present_today"] == 1
        assert result["absent_today"] == 1

    @pytest.mark.asyncio
    async def test_attendance_percentage_calculation(self, db):
        await _add_student(db, "101", "Alice")
        await _add_student(db, "102", "Bob")
        await _add_student(db, "103", "Carol")
        await _add_student(db, "104", "Dave")
        await _add_attendance(db, "101", "Alice")
        await _add_attendance(db, "102", "Bob")
        await _add_attendance(db, "103", "Carol")

        result = await analytics_service.get_dashboard_stats(db)
        assert result["attendance_percentage"] == pytest.approx(75.0)

    @pytest.mark.asyncio
    async def test_unknown_face_count_today(self, db):
        await _add_unknown_face(db)  # today
        await _add_unknown_face(db)  # today
        # Yesterday's unknown face should not be counted
        yesterday_ts = (datetime.utcnow() - timedelta(days=1)).isoformat()
        await _add_unknown_face(db, timestamp=yesterday_ts)

        result = await analytics_service.get_dashboard_stats(db)
        assert result["unknown_face_count"] == 2


# ---------------------------------------------------------------------------
# get_trends
# ---------------------------------------------------------------------------

class TestGetTrends:
    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self, db):
        result = await analytics_service.get_trends(db, "daily")
        assert result == []

    @pytest.mark.asyncio
    async def test_daily_trends_returns_date_count_dicts(self, db):
        await _add_student(db, "101", "Alice")
        await _add_attendance(db, "101", "Alice")

        result = await analytics_service.get_trends(db, "daily")
        assert len(result) >= 1
        record = result[0]
        assert "date" in record
        assert "count" in record

    @pytest.mark.asyncio
    async def test_weekly_trends_no_error(self, db):
        result = await analytics_service.get_trends(db, "weekly")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_monthly_trends_no_error(self, db):
        result = await analytics_service.get_trends(db, "monthly")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_multiple_students_same_day_count_as_one(self, db):
        today = datetime.utcnow().date().isoformat()
        await _add_student(db, "101", "Alice")
        await _add_student(db, "102", "Bob")
        await _add_attendance(db, "101", "Alice", date_str=today)
        await _add_attendance(db, "102", "Bob", date_str=today)

        result = await analytics_service.get_trends(db, "daily")
        # Find today's entry
        today_entry = next((r for r in result if r["date"] == today), None)
        assert today_entry is not None
        assert today_entry["count"] == 2


# ---------------------------------------------------------------------------
# get_department_stats
# ---------------------------------------------------------------------------

class TestGetDepartmentStats:
    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self, db):
        result = await analytics_service.get_department_stats(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_groups_by_department(self, db):
        await _add_student(db, "101", "Alice", department="CS")
        await _add_student(db, "102", "Bob", department="Math")
        await _add_student(db, "103", "Carol", department="CS")

        result = await analytics_service.get_department_stats(db)
        departments = {r["department"] for r in result}
        assert "CS" in departments
        assert "Math" in departments

    @pytest.mark.asyncio
    async def test_present_count_reflects_today_attendance(self, db):
        await _add_student(db, "101", "Alice", department="CS")
        await _add_student(db, "102", "Bob", department="CS")
        await _add_attendance(db, "101", "Alice")  # only Alice present

        result = await analytics_service.get_department_stats(db)
        cs_row = next(r for r in result if r["department"] == "CS")
        assert cs_row["total"] == 2
        assert cs_row["present"] == 1

    @pytest.mark.asyncio
    async def test_percentage_calculation(self, db):
        await _add_student(db, "101", "Alice", department="CS")
        await _add_student(db, "102", "Bob", department="CS")
        await _add_attendance(db, "101", "Alice")

        result = await analytics_service.get_department_stats(db)
        cs_row = next(r for r in result if r["department"] == "CS")
        assert cs_row["percentage"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_zero_present_percentage_is_zero(self, db):
        await _add_student(db, "101", "Alice", department="CS")

        result = await analytics_service.get_department_stats(db)
        cs_row = next(r for r in result if r["department"] == "CS")
        assert cs_row["percentage"] == 0.0


# ---------------------------------------------------------------------------
# get_student_stats
# ---------------------------------------------------------------------------

class TestGetStudentStats:
    @pytest.mark.asyncio
    async def test_raises_key_error_for_missing_student(self, db):
        with pytest.raises(KeyError):
            await analytics_service.get_student_stats(db, "999")

    @pytest.mark.asyncio
    async def test_returns_zero_present_when_no_attendance(self, db):
        await _add_student(db, "101", "Alice")
        result = await analytics_service.get_student_stats(db, "101")
        assert result["total_present"] == 0
        assert result["roll_number"] == "101"

    @pytest.mark.asyncio
    async def test_total_present_counts_distinct_dates(self, db):
        today = datetime.utcnow().date().isoformat()
        await _add_student(db, "101", "Alice")
        # Two records same date — should count as 1 distinct day
        await _add_attendance(db, "101", "Alice", date_str=today, time_str="09:00:00")
        await _add_attendance(db, "101", "Alice", date_str=today, time_str="10:00:00")

        result = await analytics_service.get_student_stats(db, "101")
        assert result["total_present"] == 1

    @pytest.mark.asyncio
    async def test_returns_records_list(self, db):
        today = datetime.utcnow().date().isoformat()
        await _add_student(db, "101", "Alice")
        await _add_attendance(db, "101", "Alice", date_str=today)

        result = await analytics_service.get_student_stats(db, "101")
        assert isinstance(result["records"], list)
        assert len(result["records"]) == 1


# ---------------------------------------------------------------------------
# get_heatmap
# ---------------------------------------------------------------------------

class TestGetHeatmap:
    @pytest.mark.asyncio
    async def test_returns_90_entries(self, db):
        await _add_student(db, "101", "Alice")
        result = await analytics_service.get_heatmap(db, "101")
        assert len(result) == 90

    @pytest.mark.asyncio
    async def test_attended_date_is_present(self, db):
        today = date.today().isoformat()
        await _add_student(db, "101", "Alice")
        await _add_attendance(db, "101", "Alice", date_str=today)

        result = await analytics_service.get_heatmap(db, "101")
        assert result[today] == "Present"

    @pytest.mark.asyncio
    async def test_unattended_date_is_absent(self, db):
        await _add_student(db, "101", "Alice")
        result = await analytics_service.get_heatmap(db, "101")
        # Today should be Absent if no attendance was recorded
        today = date.today().isoformat()
        assert result[today] == "Absent"

    @pytest.mark.asyncio
    async def test_all_keys_are_date_strings(self, db):
        await _add_student(db, "101", "Alice")
        result = await analytics_service.get_heatmap(db, "101")
        for key in result:
            # Should be parseable as ISO date
            date.fromisoformat(key)  # raises ValueError if invalid
