"""
Analytics service.

Provides aggregated query functions for the /api/analytics routes.
All queries use ``?`` parameterised placeholders — no raw string interpolation.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

async def get_dashboard_stats(db) -> Dict[str, Any]:
    """
    Return a dict with:
      - total_students        (int)
      - present_today         (int)
      - absent_today          (int)  = total_students - present_today
      - attendance_percentage (float, 0.0 when total_students == 0)
      - unknown_face_count    (int)

    Requirement 8.1
    """
    today = date.today().isoformat()

    # Total enrolled students
    row = await db.execute_fetchall("SELECT COUNT(*) AS cnt FROM students")
    total_students: int = row[0]["cnt"] if row else 0

    # Distinct students who have an attendance row for today
    row = await db.execute_fetchall(
        "SELECT COUNT(DISTINCT roll_number) AS cnt FROM attendance WHERE date = ?",
        (today,),
    )
    present_today: int = row[0]["cnt"] if row else 0

    absent_today: int = total_students - present_today

    if total_students > 0:
        attendance_percentage = round(present_today * 100.0 / total_students, 2)
    else:
        attendance_percentage = 0.0

    # Unknown face events logged today
    row = await db.execute_fetchall(
        "SELECT COUNT(*) AS cnt FROM unknown_faces WHERE DATE(timestamp) = ?",
        (today,),
    )
    unknown_face_count: int = row[0]["cnt"] if row else 0

    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": attendance_percentage,
        "unknown_face_count": unknown_face_count,
    }


# ---------------------------------------------------------------------------
# Attendance trends
# ---------------------------------------------------------------------------

async def get_trends(db, period: str = "daily") -> List[Dict[str, Any]]:
    """
    Return a list of {"date": <str>, "count": <int>} for the requested period.

    - "daily"   — last 30 days, grouped by calendar date
    - "weekly"  — last 12 weeks, grouped by ISO week (%Y-W%W)
    - "monthly" — last 12 months, grouped by year-month (%Y-%m)

    Requirement 8.2
    """
    today = date.today()

    if period == "weekly":
        start_date = (today - timedelta(weeks=12)).isoformat()
        group_expr = "strftime('%Y-W%W', date)"
    elif period == "monthly":
        start_date = (today - timedelta(days=365)).isoformat()
        group_expr = "strftime('%Y-%m', date)"
    else:
        # Default: daily — last 30 days
        start_date = (today - timedelta(days=30)).isoformat()
        group_expr = "date"

    sql = f"""
        SELECT {group_expr} AS period_label,
               COUNT(DISTINCT roll_number) AS count
        FROM attendance
        WHERE date >= ?
        GROUP BY {group_expr}
        ORDER BY {group_expr} ASC
    """
    rows = await db.execute_fetchall(sql, (start_date,))
    return [{"date": row["period_label"], "count": row["count"]} for row in rows]


# ---------------------------------------------------------------------------
# Department statistics
# ---------------------------------------------------------------------------

async def get_department_stats(db) -> List[Dict[str, Any]]:
    """
    Return a list of {department, total, present, percentage} for today.

    Uses a LEFT JOIN so departments with zero attendance still appear.

    Requirement 8.3
    """
    today = date.today().isoformat()

    sql = """
        SELECT s.department,
               COUNT(DISTINCT s.roll_number)                          AS total,
               COUNT(DISTINCT a.roll_number)                          AS present,
               ROUND(
                   COUNT(DISTINCT a.roll_number) * 100.0
                   / NULLIF(COUNT(DISTINCT s.roll_number), 0),
                   2
               ) AS percentage
        FROM students s
        LEFT JOIN attendance a
               ON a.roll_number = s.roll_number
              AND a.date = ?
        GROUP BY s.department
    """
    rows = await db.execute_fetchall(sql, (today,))
    return [
        {
            "department": row["department"],
            "total": row["total"],
            "present": row["present"],
            "percentage": row["percentage"] if row["percentage"] is not None else 0.0,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Individual student statistics
# ---------------------------------------------------------------------------

async def get_student_stats(db, roll_number: str) -> Dict[str, Any]:
    """
    Return per-student statistics:
      - roll_number
      - total_present         number of distinct dates with an attendance row
      - total_absent          working days since enrollment minus total_present
      - attendance_percentage
      - records               list of all attendance rows for this student

    ``total_absent`` is calculated as (days since created_at) - total_present,
    floored at 0 so it never goes negative.

    Requirement 8.4
    """
    # Verify student exists
    student_rows = await db.execute_fetchall(
        "SELECT roll_number, name, created_at FROM students WHERE roll_number = ?",
        (roll_number,),
    )
    if not student_rows:
        raise KeyError(f"Student '{roll_number}' not found.")

    student = student_rows[0]

    # Distinct attended days
    cnt_rows = await db.execute_fetchall(
        "SELECT COUNT(DISTINCT date) AS cnt FROM attendance WHERE roll_number = ?",
        (roll_number,),
    )
    total_present: int = cnt_rows[0]["cnt"] if cnt_rows else 0

    # Days since enrollment (inclusive of today)
    try:
        enrolled_date = date.fromisoformat(student["created_at"][:10])
        days_since_enrollment = (date.today() - enrolled_date).days + 1
    except (ValueError, TypeError):
        days_since_enrollment = total_present  # fallback: no absences

    total_absent: int = max(0, days_since_enrollment - total_present)

    total_days = total_present + total_absent
    attendance_percentage = (
        round(total_present * 100.0 / total_days, 2) if total_days > 0 else 0.0
    )

    # Full attendance records
    record_rows = await db.execute_fetchall(
        """
        SELECT id, roll_number, name, date, time,
               confidence_score, status, marked_by, created_at
        FROM attendance
        WHERE roll_number = ?
        ORDER BY date DESC, time DESC
        """,
        (roll_number,),
    )
    records = [dict(row) for row in record_rows]

    return {
        "roll_number": roll_number,
        "total_present": total_present,
        "total_absent": total_absent,
        "attendance_percentage": attendance_percentage,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Attendance heatmap
# ---------------------------------------------------------------------------

async def get_heatmap(db, roll_number: str) -> Dict[str, str]:
    """
    Return a mapping of date strings → "Present" | "Absent" for the last
    90 calendar days.

    Every day in the window is included; days with no attendance row are
    labelled "Absent".

    Requirement 8.5
    """
    today = date.today()
    start = today - timedelta(days=89)  # 90 days inclusive

    # Fetch all attendance dates for this student in the window
    rows = await db.execute_fetchall(
        """
        SELECT DISTINCT date
        FROM attendance
        WHERE roll_number = ? AND date >= ?
        """,
        (roll_number, start.isoformat()),
    )
    attended_dates = {row["date"] for row in rows}

    heatmap: Dict[str, str] = {}
    current = start
    while current <= today:
        iso = current.isoformat()
        heatmap[iso] = "Present" if iso in attended_dates else "Absent"
        current += timedelta(days=1)

    return heatmap
