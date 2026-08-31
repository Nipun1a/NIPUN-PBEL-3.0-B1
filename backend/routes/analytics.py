"""
routes/analytics.py

FastAPI APIRouter for all analytics endpoints.

Prefix (/api/analytics) is applied externally in main.py.

Endpoints:
  GET /dashboard              → dashboard stats (totals, attendance %, unknown faces)
  GET /trends                 → time-series attendance; ?period=daily|weekly|monthly
  GET /department             → per-department attendance counts and percentages
  GET /student/{roll_number}  → individual student stats (404 if student not found)
  GET /heatmap                → 90-day date-keyed attendance status; ?roll_number=<rn>

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

from typing import List

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.database.connection import get_db
from backend.models.analytics import DashboardStats, DepartmentStats, TrendData
from backend.services import analytics_service

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /dashboard  → dashboard stats
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return high-level attendance statistics for today.

    Response fields:
      - total_students        — total enrolled students
      - present_today         — distinct students marked present today
      - absent_today          — total_students minus present_today
      - attendance_percentage — (present / total) × 100, rounded to 2 dp
      - unknown_face_count    — unknown face events logged today

    Requirements: 8.1
    """
    stats = await analytics_service.get_dashboard_stats(db)
    return stats


# ---------------------------------------------------------------------------
# GET /trends  → time-series attendance data
# ---------------------------------------------------------------------------

@router.get("/trends", response_model=List[TrendData])
async def get_trends(
    period: str = Query(
        default="daily",
        description="Aggregation period: 'daily' (last 30 days), "
                    "'weekly' (last 12 weeks), or 'monthly' (last 12 months)",
    ),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return time-series attendance counts grouped by the requested period.

    Each item in the list has:
      - date  — the period label (YYYY-MM-DD for daily, YYYY-Www for weekly,
                 YYYY-MM for monthly)
      - count — distinct students present in that period

    Unknown ``period`` values fall back to "daily" behaviour inside the
    service layer.

    Requirements: 8.2
    """
    trends = await analytics_service.get_trends(db, period=period)
    return trends


# ---------------------------------------------------------------------------
# GET /department  → department-wise attendance
# ---------------------------------------------------------------------------

@router.get("/department", response_model=List[DepartmentStats])
async def get_department_stats(
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return today's attendance broken down by department.

    Each item in the list has:
      - department  — department name (empty string if not set on student)
      - total       — total enrolled students in this department
      - present     — students present today in this department
      - percentage  — (present / total) × 100, rounded to 2 dp; 0.0 when total is 0

    Departments with zero attendance today are still included (LEFT JOIN).

    Requirements: 8.3
    """
    dept_stats = await analytics_service.get_department_stats(db)
    return dept_stats


# ---------------------------------------------------------------------------
# GET /student/{roll_number}  → individual student stats
# ---------------------------------------------------------------------------

@router.get("/student/{roll_number}")
async def get_student_stats(
    roll_number: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return detailed attendance statistics for a single student.

    Response fields:
      - roll_number           — student identifier
      - total_present         — distinct calendar dates with an attendance record
      - total_absent          — days since enrollment minus total_present (min 0)
      - attendance_percentage — (present / total_days) × 100, rounded to 2 dp
      - records               — full list of attendance rows, newest first

    Returns HTTP 404 when the roll_number is not found in the students table.

    Requirements: 8.4
    """
    try:
        stats = await analytics_service.get_student_stats(db, roll_number)
        return stats
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# GET /heatmap  → 90-day attendance status map
# ---------------------------------------------------------------------------

@router.get("/heatmap")
async def get_heatmap(
    roll_number: str = Query(
        ...,
        description="Roll number of the student whose heatmap is requested",
    ),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a 90-day calendar heatmap for the specified student.

    The response is a JSON object whose keys are ISO date strings
    (YYYY-MM-DD) and whose values are either "Present" or "Absent".
    Every day in the 90-day window is included.

    Example::

        {
          "2024-01-01": "Present",
          "2024-01-02": "Absent",
          ...
        }

    Requirements: 8.5
    """
    try:
        heatmap = await analytics_service.get_heatmap(db, roll_number)
        return heatmap
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
