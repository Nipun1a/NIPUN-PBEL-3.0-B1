"""
routes/attendance.py

FastAPI APIRouter for all attendance-related endpoints.

Prefix (/api/attendance) is applied externally in main.py.

Endpoints:
  GET    /today           → all attendance records for today (Req 6.2)
  GET    /               → paginated + filtered attendance records (Req 6.1)
  PUT    /{id}           → update an attendance record (Req 6.3)
  DELETE /{id}           → delete an attendance record (Req 6.4)
  POST   /manual         → create a manual attendance entry (Req 6.5, 6.6)

NOTE: GET /today is declared before GET /{id} (if any) to prevent FastAPI
      from treating the literal path segment "today" as an integer id.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""
from __future__ import annotations

from typing import Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.models.attendance import ManualAttendance
from backend.services import attendance_service


class AttendanceUpdate(BaseModel):
    """Partial update payload for PUT /{id}. All fields are optional."""

    roll_number: Optional[str] = None
    name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    confidence_score: Optional[float] = None
    status: Optional[str] = None
    marked_by: Optional[str] = None

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /today  → all attendance records for today
# NOTE: MUST be declared before any /{id} route to avoid route conflicts.
# ---------------------------------------------------------------------------

@router.get("/today")
async def get_today_attendance(
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return all attendance records for the current UTC date.

    Response: list of attendance record objects ordered by time ascending.

    Requirements: 6.2
    """
    records = await attendance_service.get_today(db)
    return records


# ---------------------------------------------------------------------------
# GET /  → paginated + filtered attendance records
# ---------------------------------------------------------------------------

@router.get("/")
async def get_attendance(
    date: Optional[str] = Query(default=None, description="Filter by date (YYYY-MM-DD)"),
    roll_number: Optional[str] = Query(default=None, description="Filter by roll number"),
    name: Optional[str] = Query(default=None, description="Filter by student name (partial match)"),
    status: Optional[str] = Query(default=None, description="Filter by status (e.g. 'Present', 'Absent')"),
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    page_size: int = Query(default=50, ge=1, le=500, description="Records per page"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a paginated, optionally-filtered list of attendance records.

    All filter parameters are optional and combined with AND logic.
    The ``name`` filter performs a case-insensitive partial (LIKE) match.

    Response shape::

        {
            "total": <int>,
            "page": <int>,
            "page_size": <int>,
            "records": [ /* AttendanceRecord objects */ ]
        }

    Requirements: 6.1
    """
    return await attendance_service.get_filtered(
        db,
        date=date,
        roll_number=roll_number,
        name=name,
        status=status,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# PUT /{id}  → update an attendance record
# ---------------------------------------------------------------------------

@router.put("/{record_id}")
async def update_attendance(
    record_id: int,
    payload: AttendanceUpdate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Update fields on an existing attendance record identified by *record_id*.

    Only fields present in the request body are applied (partial update).
    Returns 404 if the record does not exist.

    Requirements: 6.3
    """
    try:
        updated = await attendance_service.update(
            record_id,
            payload.model_dump(exclude_unset=True),
            db,
        )
        return updated
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /{id}  → delete an attendance record
# ---------------------------------------------------------------------------

@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_attendance(
    record_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete the attendance record with *record_id*.

    Returns 404 if the record does not exist.
    Returns 204 No Content on success.

    Requirements: 6.4
    """
    try:
        await attendance_service.delete(record_id, db)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# POST /manual  → create a manual attendance entry
# NOTE: MUST be declared before /{id} routes (or at least not conflict).
#       Since /manual is a fixed string and /{id} expects an int, FastAPI
#       resolves these correctly — but declaring /manual early is good practice.
# ---------------------------------------------------------------------------

@router.post("/manual", status_code=status.HTTP_201_CREATED)
async def create_manual_attendance(
    payload: ManualAttendance,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Create a manual attendance entry for a student.

    The roll_number must correspond to an existing student in the students
    table. Returns 404 if the student is not found.

    Returns the newly created attendance record with HTTP 201.

    Requirements: 6.5, 6.6
    """
    try:
        record = await attendance_service.create_manual(
            roll_number=payload.roll_number,
            date=payload.date,
            time=payload.time,
            status=payload.status,
            db=db,
        )
        return record
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
