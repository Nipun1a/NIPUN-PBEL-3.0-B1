"""
Export service for generating Excel workbooks using openpyxl.

Provides in-memory workbook generation for attendance records and student lists.
Returns raw bytes suitable for streaming as an HTTP response.
"""

import openpyxl
from io import BytesIO


def build_attendance_workbook(records: list) -> bytes:
    """
    Build an Excel workbook from a list of attendance record dicts.

    Each record dict is expected to have the keys:
        roll_number, name, department, date, time,
        confidence_score, status, marked_by

    If records is empty, the workbook is returned with headers only.

    Args:
        records: List of attendance record dicts.

    Returns:
        Raw bytes of the .xlsx workbook.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance"

    headers = [
        "Roll Number",
        "Name",
        "Department",
        "Date",
        "Time",
        "Confidence Score",
        "Status",
        "Marked By",
    ]
    ws.append(headers)

    for rec in records:
        ws.append([
            rec["roll_number"],
            rec["name"],
            rec.get("department", ""),
            rec["date"],
            rec["time"],
            rec["confidence_score"],
            rec["status"],
            rec["marked_by"],
        ])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_students_workbook(students: list) -> bytes:
    """
    Build an Excel workbook from a list of student dicts.

    Each student dict is expected to have the keys:
        roll_number, name, department, email, phone, created_at

    If students is empty, the workbook is returned with headers only.

    Args:
        students: List of student dicts.

    Returns:
        Raw bytes of the .xlsx workbook.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Students"

    headers = [
        "Roll Number",
        "Name",
        "Department",
        "Email",
        "Phone",
        "Created At",
    ]
    ws.append(headers)

    for student in students:
        ws.append([
            student["roll_number"],
            student["name"],
            student.get("department", ""),
            student.get("email", ""),
            student.get("phone", ""),
            student.get("created_at", ""),
        ])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
