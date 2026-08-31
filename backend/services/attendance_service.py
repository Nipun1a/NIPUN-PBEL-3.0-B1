"""
Attendance service — all async database operations for the attendance table.

All queries use parameterised ? placeholders exclusively (Requirement 20.5).

Covered requirements:
  6.1 – Paginated + filterable attendance query
  6.2 – Today's attendance query
  6.3 – Update attendance record
  6.4 – Delete attendance record
  6.5 – Manual attendance entry with marked_by='manual'
  18.1 – Duplicate suppression within cooldown window
  18.2 – Insert new record after cooldown has elapsed
  18.4 – Duplicate check enforced at service layer before DB write
  20.5 – Parameterised queries only
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Duplicate prevention
# ---------------------------------------------------------------------------

async def is_duplicate(roll_number: str, cooldown_seconds: int, db) -> bool:
    """
    Return True if an attendance record for *roll_number* exists on today's
    date with a ``created_at`` timestamp within the cooldown window.

    Requirements 18.1, 18.4 — duplicate check at service layer.
    """
    today = datetime.utcnow().date().isoformat()
    cutoff = (datetime.utcnow() - timedelta(seconds=cooldown_seconds)).isoformat()
    row = await db.execute_fetchone(
        "SELECT id FROM attendance "
        "WHERE roll_number = ? AND date = ? AND created_at > ?",
        (roll_number, today, cutoff),
    )
    return row is not None


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

async def mark_attendance(
    roll_number: str,
    name: str,
    confidence: float,
    db,
) -> int:
    """
    Insert a new attendance row marked by face recognition and return its id.

    Requirements 6.1, 18.2.
    """
    now = datetime.utcnow()
    cursor = await db.execute(
        """INSERT INTO attendance
               (roll_number, name, date, time, confidence_score, status, marked_by, created_at)
           VALUES (?, ?, ?, ?, ?, 'Present', 'face_recognition', ?)""",
        (
            roll_number,
            name,
            now.date().isoformat(),
            now.strftime("%H:%M:%S"),
            confidence,
            now.isoformat(),
        ),
    )
    await db.commit()
    return cursor.lastrowid


async def create_manual(
    roll_number: str,
    date: str,
    time: str,
    status: str,
    db,
) -> Dict[str, Any]:
    """
    Insert a manually-added attendance record.

    Raises ``KeyError`` if *roll_number* is not present in the students table
    (Requirement 6.5, 6.6).

    Returns the newly created record as a dict.
    """
    # Verify student exists before writing attendance.
    student_row = await db.execute_fetchone(
        "SELECT name FROM students WHERE roll_number = ?",
        (roll_number,),
    )
    if student_row is None:
        raise KeyError(f"Student with roll_number '{roll_number}' not found")

    name = student_row["name"]
    now = datetime.utcnow().isoformat()

    cursor = await db.execute(
        """INSERT INTO attendance
               (roll_number, name, date, time, confidence_score, status, marked_by, created_at)
           VALUES (?, ?, ?, ?, 0.0, ?, 'manual', ?)""",
        (roll_number, name, date, time, status, now),
    )
    await db.commit()
    row_id = cursor.lastrowid

    # Return the freshly inserted record.
    row = await db.execute_fetchone(
        "SELECT * FROM attendance WHERE id = ?",
        (row_id,),
    )
    return dict(row)


async def update(record_id: int, updates: Dict[str, Any], db) -> Dict[str, Any]:
    """
    Apply *updates* to the attendance row identified by *record_id*.

    Raises ``KeyError`` if the record does not exist (Requirement 6.3).
    Returns the updated record as a dict.
    """
    # Confirm row exists.
    existing = await db.execute_fetchone(
        "SELECT * FROM attendance WHERE id = ?",
        (record_id,),
    )
    if existing is None:
        raise KeyError(f"Attendance record with id {record_id} not found")

    if not updates:
        return dict(existing)

    # Build a dynamic SET clause from the provided fields.
    allowed_fields = {"roll_number", "name", "date", "time",
                      "confidence_score", "status", "marked_by"}
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}

    if not filtered:
        return dict(existing)

    set_clause = ", ".join(f"{col} = ?" for col in filtered)
    values = list(filtered.values()) + [record_id]

    await db.execute(
        f"UPDATE attendance SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await db.commit()

    updated_row = await db.execute_fetchone(
        "SELECT * FROM attendance WHERE id = ?",
        (record_id,),
    )
    return dict(updated_row)


async def delete(record_id: int, db) -> None:
    """
    Delete the attendance row with *record_id*.

    Raises ``KeyError`` if the record does not exist (Requirement 6.4).
    """
    existing = await db.execute_fetchone(
        "SELECT id FROM attendance WHERE id = ?",
        (record_id,),
    )
    if existing is None:
        raise KeyError(f"Attendance record with id {record_id} not found")

    await db.execute(
        "DELETE FROM attendance WHERE id = ?",
        (record_id,),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

async def get_today(db) -> List[Dict[str, Any]]:
    """
    Return all attendance records for the current UTC date.

    Requirement 6.2.
    """
    today = datetime.utcnow().date().isoformat()
    rows = await db.execute_fetchall(
        "SELECT * FROM attendance WHERE date = ? ORDER BY time ASC",
        (today,),
    )
    return [dict(r) for r in rows]


async def get_filtered(
    db,
    date: Optional[str] = None,
    roll_number: Optional[str] = None,
    name: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    """
    Return a paginated, optionally-filtered list of attendance records.

    All filters use parameterised ``?`` placeholders (Requirement 20.5).
    Returns a dict with keys: ``total``, ``page``, ``page_size``, ``records``.

    Requirement 6.1.
    """
    conditions: List[str] = []
    params: List[Any] = []

    if date is not None:
        conditions.append("date = ?")
        params.append(date)

    if roll_number is not None:
        conditions.append("roll_number = ?")
        params.append(roll_number)

    if name is not None:
        conditions.append("name LIKE ?")
        params.append(f"%{name}%")

    if status is not None:
        conditions.append("status = ?")
        params.append(status)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Count total matching records.
    count_row = await db.execute_fetchone(
        f"SELECT COUNT(*) AS total FROM attendance {where_clause}",  # noqa: S608
        params,
    )
    total: int = count_row["total"] if count_row else 0

    # Clamp page to valid range.
    page = max(1, page)
    offset = (page - 1) * page_size

    rows = await db.execute_fetchall(
        f"SELECT * FROM attendance {where_clause} "  # noqa: S608
        f"ORDER BY date DESC, time DESC LIMIT ? OFFSET ?",
        params + [page_size, offset],
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": [dict(r) for r in rows],
    }
