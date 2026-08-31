"""
routes/export.py

FastAPI APIRouter for Excel export endpoints.

Prefix (/api/export) is applied externally in main.py.

Endpoints:
  GET /attendance  → Export filtered attendance records as .xlsx
  GET /students    → Export all students as .xlsx

Query parameters for /attendance:
  date         – exact date filter (YYYY-MM-DD)
  start_date   – inclusive lower bound (YYYY-MM-DD)
  end_date     – inclusive upper bound (YYYY-MM-DD)
  roll_number  – exact roll_number filter
  department   – exact department filter (joined from students table)

All filters are optional; if none are provided all records are exported.
Empty result sets produce a headers-only workbook (not an error).

Requirements: 7.1, 7.2, 7.3, 7.4
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from backend.database.connection import get_db
from backend.services import attendance_service, export_service, student_service

router = APIRouter()

_XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ---------------------------------------------------------------------------
# GET /attendance  → Export attendance records as xlsx
# ---------------------------------------------------------------------------

@router.get("/attendance")
async def export_attendance(
    date: Optional[str] = Query(
        default=None,
        description="Exact date filter (YYYY-MM-DD)",
    ),
    start_date: Optional[str] = Query(
        default=None,
        description="Inclusive start date for date-range filter (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="Inclusive end date for date-range filter (YYYY-MM-DD)",
    ),
    roll_number: Optional[str] = Query(
        default=None,
        description="Filter by exact student roll number",
    ),
    department: Optional[str] = Query(
        default=None,
        description="Filter by exact student department",
    ),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Export attendance records as an Excel workbook.

    Supports optional filters: date, start_date, end_date, roll_number,
    department.  When no records match the filters, a headers-only workbook
    is returned rather than an error.

    Response:
      Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
      Content-Disposition: attachment; filename="attendance_export.xlsx"

    Requirements: 7.1, 7.2, 7.3
    """
    records = await _fetch_attendance_records(
        db,
        date=date,
        start_date=start_date,
        end_date=end_date,
        roll_number=roll_number,
        department=department,
    )

    xlsx_bytes = export_service.build_attendance_workbook(records)

    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="attendance_export.xlsx"'},
    )


# ---------------------------------------------------------------------------
# GET /students  → Export all students as xlsx
# ---------------------------------------------------------------------------

@router.get("/students")
async def export_students(
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Export all students as an Excel workbook.

    Response:
      Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
      Content-Disposition: attachment; filename="students_export.xlsx"

    Requirements: 7.1, 7.4
    """
    result = await student_service.list_students(db, page=1, page_size=10_000)
    students = result["records"]

    xlsx_bytes = export_service.build_students_workbook(students)

    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="students_export.xlsx"'},
    )


# ---------------------------------------------------------------------------
# Private helper — fetch and filter attendance with department join
# ---------------------------------------------------------------------------

async def _fetch_attendance_records(
    db: aiosqlite.Connection,
    *,
    date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    roll_number: Optional[str],
    department: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Fetch attendance records with optional filters.

    When `department` is provided, a LEFT JOIN against the students table is
    used so that the column is available for filtering.  The attendance row's
    `name` and `roll_number` columns are used directly; `department` is pulled
    from the students table.

    All filter values are passed as parameterised query arguments — no string
    interpolation (Requirements 20.5).
    """
    conditions: List[str] = []
    params: List[Any] = []

    if date is not None:
        conditions.append("a.date = ?")
        params.append(date)
    else:
        if start_date is not None:
            conditions.append("a.date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("a.date <= ?")
            params.append(end_date)

    if roll_number is not None:
        conditions.append("a.roll_number = ?")
        params.append(roll_number)

    if department is not None:
        conditions.append("s.department = ?")
        params.append(department)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = (
        "SELECT a.roll_number, a.name, "
        "COALESCE(s.department, '') AS department, "
        "a.date, a.time, a.confidence_score, a.status, a.marked_by "
        "FROM attendance a "
        "LEFT JOIN students s ON a.roll_number = s.roll_number "
        f"{where_clause} "  # noqa: S608 — where_clause contains no user input
        "ORDER BY a.date DESC, a.time DESC"
    )

    rows = await db.execute_fetchall(sql, params)
    return [dict(r) for r in rows]
