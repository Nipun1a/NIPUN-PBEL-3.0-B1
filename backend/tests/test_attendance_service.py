"""
Unit tests for backend/services/attendance_service.py.

Tests:
  - is_duplicate: returns False on empty DB
  - is_duplicate: returns True when record exists within cooldown
  - is_duplicate: returns False when record exists but outside cooldown
  - is_duplicate: boundary — record exactly at cutoff boundary
  - mark_attendance: inserts correct row, returns integer id
  - mark_attendance: today's date and time are recorded
  - get_today: returns only today's records
  - get_filtered: pagination works
  - get_filtered: filters by date, roll_number, name, status
  - update: updates specified fields, returns updated record
  - update: raises KeyError for missing record
  - delete: deletes existing record
  - delete: raises KeyError for missing record
  - create_manual: inserts record with marked_by='manual'
  - create_manual: raises KeyError when student not found
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import aiosqlite

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.tests.conftest import _setup_db, _patch_db
from backend.services import attendance_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert_student(db, roll_number="101", name="Alice"):
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO students (roll_number, name, department, email, phone, created_at, updated_at)"
        " VALUES (?, ?, '', '', ?, ?, ?)",
        (roll_number, name, now, now, now),
    )
    await db.commit()


async def _insert_attendance(
    db,
    roll_number="101",
    name="Alice",
    date=None,
    time="09:00:00",
    created_at=None,
):
    if date is None:
        date = datetime.utcnow().date().isoformat()
    if created_at is None:
        created_at = datetime.utcnow().isoformat()

    await db.execute(
        """INSERT INTO attendance
               (roll_number, name, date, time, confidence_score, status, marked_by, created_at)
           VALUES (?, ?, ?, ?, 0.9, 'Present', 'face_recognition', ?)""",
        (roll_number, name, date, time, created_at),
    )
    await db.commit()


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
# is_duplicate tests
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    @pytest.mark.asyncio
    async def test_returns_false_on_empty_db(self, db):
        result = await attendance_service.is_duplicate("101", 300, db)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_within_cooldown(self, db):
        # Insert a record created 60 seconds ago (within 300s cooldown)
        created_at = (datetime.utcnow() - timedelta(seconds=60)).isoformat()
        await _insert_attendance(db, roll_number="101", created_at=created_at)

        result = await attendance_service.is_duplicate("101", 300, db)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_outside_cooldown(self, db):
        # Insert a record created 400 seconds ago (outside 300s cooldown)
        created_at = (datetime.utcnow() - timedelta(seconds=400)).isoformat()
        await _insert_attendance(db, roll_number="101", created_at=created_at)

        result = await attendance_service.is_duplicate("101", 300, db)
        assert result is False

    @pytest.mark.asyncio
    async def test_different_roll_number_not_duplicate(self, db):
        # Record for roll_number "202" should not affect "101"
        created_at = (datetime.utcnow() - timedelta(seconds=10)).isoformat()
        await _insert_attendance(db, roll_number="202", created_at=created_at)

        result = await attendance_service.is_duplicate("101", 300, db)
        assert result is False

    @pytest.mark.asyncio
    async def test_different_date_not_duplicate(self, db):
        # Record for yesterday should not be a duplicate for today's check
        yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
        created_at = (datetime.utcnow() - timedelta(days=1)).isoformat()
        await _insert_attendance(db, date=yesterday, created_at=created_at)

        result = await attendance_service.is_duplicate("101", 300, db)
        assert result is False

    @pytest.mark.asyncio
    async def test_zero_cooldown_always_false(self, db):
        # With cooldown=0, cutoff == now, so no record can be "within" the window
        created_at = (datetime.utcnow() - timedelta(seconds=1)).isoformat()
        await _insert_attendance(db, roll_number="101", created_at=created_at)

        result = await attendance_service.is_duplicate("101", 0, db)
        assert result is False


# ---------------------------------------------------------------------------
# mark_attendance tests
# ---------------------------------------------------------------------------

class TestMarkAttendance:
    @pytest.mark.asyncio
    async def test_returns_integer_id(self, db):
        await _insert_student(db, "101", "Alice")
        row_id = await attendance_service.mark_attendance("101", "Alice", 0.85, db)
        assert isinstance(row_id, int)
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_inserts_correct_row(self, db):
        await _insert_student(db, "101", "Alice")
        row_id = await attendance_service.mark_attendance("101", "Alice", 0.85, db)

        async with db.execute("SELECT * FROM attendance WHERE id = ?", (row_id,)) as cur:
            row = await cur.fetchone()

        assert row["roll_number"] == "101"
        assert row["name"] == "Alice"
        assert row["confidence_score"] == pytest.approx(0.85)
        assert row["status"] == "Present"
        assert row["marked_by"] == "face_recognition"

    @pytest.mark.asyncio
    async def test_records_todays_date(self, db):
        await _insert_student(db, "101", "Alice")
        row_id = await attendance_service.mark_attendance("101", "Alice", 0.9, db)

        today = datetime.utcnow().date().isoformat()
        async with db.execute("SELECT date FROM attendance WHERE id = ?", (row_id,)) as cur:
            row = await cur.fetchone()

        assert row["date"] == today

    @pytest.mark.asyncio
    async def test_multiple_marks_return_different_ids(self, db):
        await _insert_student(db, "101", "Alice")
        id1 = await attendance_service.mark_attendance("101", "Alice", 0.8, db)
        id2 = await attendance_service.mark_attendance("101", "Alice", 0.9, db)
        assert id1 != id2


# ---------------------------------------------------------------------------
# get_today tests
# ---------------------------------------------------------------------------

class TestGetToday:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_records(self, db):
        result = await attendance_service.get_today(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_todays_records(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_attendance(db, roll_number="101", name="Alice")

        result = await attendance_service.get_today(db)
        assert len(result) == 1
        assert result[0]["roll_number"] == "101"

    @pytest.mark.asyncio
    async def test_excludes_past_dates(self, db):
        yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
        created_at = (datetime.utcnow() - timedelta(days=1)).isoformat()
        await _insert_attendance(db, date=yesterday, created_at=created_at)

        result = await attendance_service.get_today(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_multiple_records(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_student(db, "102", "Bob")
        await _insert_attendance(db, roll_number="101", name="Alice")
        await _insert_attendance(db, roll_number="102", name="Bob")

        result = await attendance_service.get_today(db)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_filtered tests
# ---------------------------------------------------------------------------

class TestGetFiltered:
    @pytest.mark.asyncio
    async def test_returns_all_when_no_filters(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_attendance(db, roll_number="101", name="Alice")

        result = await attendance_service.get_filtered(db)
        assert result["total"] == 1
        assert len(result["records"]) == 1

    @pytest.mark.asyncio
    async def test_pagination_structure(self, db):
        result = await attendance_service.get_filtered(db, page=1, page_size=10)
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "records" in result
        assert result["page"] == 1
        assert result["page_size"] == 10

    @pytest.mark.asyncio
    async def test_filter_by_roll_number(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_student(db, "102", "Bob")
        await _insert_attendance(db, roll_number="101", name="Alice")
        await _insert_attendance(db, roll_number="102", name="Bob")

        result = await attendance_service.get_filtered(db, roll_number="101")
        assert result["total"] == 1
        assert result["records"][0]["roll_number"] == "101"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, db):
        await _insert_student(db, "101", "Alice")
        await db.execute(
            "INSERT INTO attendance (roll_number, name, date, time, confidence_score, status, marked_by, created_at)"
            " VALUES (?, ?, ?, ?, 0.0, 'Absent', 'manual', ?)",
            ("101", "Alice", datetime.utcnow().date().isoformat(), "09:00:00",
             datetime.utcnow().isoformat()),
        )
        await _insert_attendance(db, roll_number="101", name="Alice")
        await db.commit()

        result = await attendance_service.get_filtered(db, status="Absent")
        assert result["total"] == 1
        assert result["records"][0]["status"] == "Absent"

    @pytest.mark.asyncio
    async def test_empty_result_when_no_match(self, db):
        result = await attendance_service.get_filtered(db, date="2000-01-01")
        assert result["total"] == 0
        assert result["records"] == []

    @pytest.mark.asyncio
    async def test_pagination_limits_records(self, db):
        await _insert_student(db, "101", "Alice")
        # Insert 5 records
        for i in range(5):
            await _insert_attendance(db, roll_number="101", name="Alice",
                                     time=f"0{i}:00:00",
                                     created_at=(datetime.utcnow() - timedelta(hours=i)).isoformat())

        result = await attendance_service.get_filtered(db, page=1, page_size=3)
        assert result["total"] == 5
        assert len(result["records"]) == 3

    @pytest.mark.asyncio
    async def test_page_2_returns_remaining(self, db):
        await _insert_student(db, "101", "Alice")
        for i in range(5):
            await _insert_attendance(db, roll_number="101", name="Alice",
                                     time=f"0{i}:00:00",
                                     created_at=(datetime.utcnow() - timedelta(hours=i)).isoformat())

        result = await attendance_service.get_filtered(db, page=2, page_size=3)
        assert result["total"] == 5
        assert len(result["records"]) == 2


# ---------------------------------------------------------------------------
# update tests
# ---------------------------------------------------------------------------

class TestUpdate:
    @pytest.mark.asyncio
    async def test_updates_specified_fields(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_attendance(db, roll_number="101", name="Alice")

        async with db.execute("SELECT id FROM attendance LIMIT 1") as cur:
            row = await cur.fetchone()
        record_id = row["id"]

        updated = await attendance_service.update(record_id, {"status": "Absent"}, db)
        assert updated["status"] == "Absent"
        assert updated["id"] == record_id

    @pytest.mark.asyncio
    async def test_raises_key_error_for_missing_record(self, db):
        with pytest.raises(KeyError):
            await attendance_service.update(9999, {"status": "Absent"}, db)

    @pytest.mark.asyncio
    async def test_empty_updates_returns_existing_record(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_attendance(db, roll_number="101", name="Alice")

        async with db.execute("SELECT id FROM attendance LIMIT 1") as cur:
            row = await cur.fetchone()
        record_id = row["id"]

        result = await attendance_service.update(record_id, {}, db)
        assert result["id"] == record_id


# ---------------------------------------------------------------------------
# delete tests
# ---------------------------------------------------------------------------

class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_existing_record(self, db):
        await _insert_student(db, "101", "Alice")
        await _insert_attendance(db, roll_number="101", name="Alice")

        async with db.execute("SELECT id FROM attendance LIMIT 1") as cur:
            row = await cur.fetchone()
        record_id = row["id"]

        await attendance_service.delete(record_id, db)

        async with db.execute("SELECT id FROM attendance WHERE id = ?", (record_id,)) as cur:
            result = await cur.fetchone()
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_key_error_for_missing_record(self, db):
        with pytest.raises(KeyError):
            await attendance_service.delete(9999, db)


# ---------------------------------------------------------------------------
# create_manual tests
# ---------------------------------------------------------------------------

class TestCreateManual:
    @pytest.mark.asyncio
    async def test_creates_record_with_manual_marked_by(self, db):
        await _insert_student(db, "101", "Alice")
        record = await attendance_service.create_manual(
            "101", "2024-01-15", "09:00:00", "Present", db
        )
        assert record["marked_by"] == "manual"
        assert record["roll_number"] == "101"
        assert record["name"] == "Alice"
        assert record["status"] == "Present"

    @pytest.mark.asyncio
    async def test_raises_key_error_when_student_not_found(self, db):
        with pytest.raises(KeyError):
            await attendance_service.create_manual(
                "999", "2024-01-15", "09:00:00", "Present", db
            )

    @pytest.mark.asyncio
    async def test_returns_dict_with_all_fields(self, db):
        await _insert_student(db, "101", "Alice")
        record = await attendance_service.create_manual(
            "101", "2024-01-15", "09:00:00", "Present", db
        )
        required_fields = {"id", "roll_number", "name", "date", "time",
                           "confidence_score", "status", "marked_by", "created_at"}
        assert required_fields.issubset(record.keys())
