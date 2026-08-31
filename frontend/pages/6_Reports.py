"""
Reports page — AI Attendance Monitoring System

Sections:
  • Filter controls: date range, department, roll number
  • Attendance preview table (first 50 rows from /api/attendance)
  • Download Attendance Report (xlsx export from /api/export/attendance)
  • Download Student List (xlsx export from /api/export/students)
  • Summary statistics: daily / weekly / monthly counts from /api/analytics/trends

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from components.data_table import data_table  # noqa: E402
from utils.api_client import APIClient  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reports — Attendance Monitor",
    page_icon="📊",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Column rename map for display ─────────────────────────────────────────────
_RENAME_MAP = {
    "roll_number":      "Roll Number",
    "name":             "Name",
    "department":       "Department",
    "date":             "Date",
    "time":             "Time",
    "confidence_score": "Confidence Score",
    "status":           "Status",
    "marked_by":        "Marked By",
}

_DISPLAY_COLS = [
    "Roll Number", "Name", "Department", "Date",
    "Time", "Confidence Score", "Status", "Marked By",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_filters(
    start_date: date | None,
    end_date: date | None,
    department: str,
    roll_number: str,
) -> dict:
    """Build a params dict, omitting keys that have no value."""
    params: dict = {}
    if start_date:
        params["start_date"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        params["end_date"] = end_date.strftime("%Y-%m-%d")
    if department.strip():
        params["department"] = department.strip()
    if roll_number.strip():
        params["roll_number"] = roll_number.strip()
    return params


def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert raw API records list to a display-ready DataFrame."""
    if not records:
        return pd.DataFrame(columns=_DISPLAY_COLS)
    df = pd.DataFrame(records)
    df = df.rename(columns={k: v for k, v in _RENAME_MAP.items() if k in df.columns})
    keep = [c for c in _DISPLAY_COLS if c in df.columns]
    return df[keep]


def fetch_attendance_preview(filters: dict) -> list[dict]:
    """Fetch up to 50 attendance records from /api/attendance with optional filters."""
    params = {**filters, "page": 1, "page_size": 50}
    try:
        resp = api.get("/api/attendance", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data[:50]
        return data.get("records", [])[:50]
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend. Ensure the server is running on http://localhost:8000.")
        return []
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error {exc.response.status_code}: {detail}")
        return []
    except Exception as exc:
        st.error(f"Unexpected error fetching preview: {exc}")
        return []


def fetch_trends(period: str) -> list[dict]:
    """Fetch attendance trend data for a given period (daily/weekly/monthly)."""
    try:
        resp = api.get("/api/analytics/trends", params={"period": period})
        resp.raise_for_status()
        data = resp.json()
        # API may return list or dict with 'data' key
        if isinstance(data, list):
            return data
        return data.get("data", data.get("trends", []))
    except requests.exceptions.ConnectionError:
        return []
    except requests.exceptions.HTTPError:
        return []
    except Exception:
        return []


def sum_trend_count(trends: list[dict]) -> int:
    """Sum the 'count' values across all trend entries."""
    return sum(int(entry.get("count", 0)) for entry in trends)


# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Reports")
st.caption("Filter attendance data, preview records, download exports, and review trend statistics.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Filter Controls
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Filter Controls")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
        key="report_start_date",
        help="Filter records on or after this date",
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        key="report_end_date",
        help="Filter records on or before this date",
    )

with col3:
    department_input = st.text_input(
        "Department",
        placeholder="e.g. CS, ECE",
        key="report_department",
        help="Filter by department name (partial match supported)",
    )

with col4:
    roll_number_input = st.text_input(
        "Roll Number",
        placeholder="e.g. 101",
        key="report_roll_number",
        help="Filter by student roll number",
    )

# Validate date range
if start_date and end_date and start_date > end_date:
    st.warning("⚠️ Start Date is after End Date. Please adjust the date range.")
    start_date_valid = False
else:
    start_date_valid = True

# Build the filters dict for API calls
active_filters = build_filters(
    start_date if start_date_valid else None,
    end_date if start_date_valid else None,
    department_input,
    roll_number_input,
)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Attendance Preview Table
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Attendance Preview")
st.caption("Showing up to 50 records matching the active filters.")

with st.spinner("Loading attendance records…"):
    preview_records = fetch_attendance_preview(active_filters)

if preview_records:
    st.info(f"Found **{len(preview_records)}** record(s) (capped at 50).")

df_preview = records_to_df(preview_records)
data_table(df_preview, key="reports_preview_table")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Downloads
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("⬇️ Downloads")
st.caption("Export data to Excel. The attendance export uses the current filter settings.")

dl_col1, dl_col2 = st.columns(2)

# ── Download Attendance Report ────────────────────────────────────────────────
with dl_col1:
    st.markdown("**Attendance Report**")
    st.caption(
        "Exports all attendance records matching the current filters "
        "(date range, department, roll number) to an Excel file."
    )
    try:
        attendance_bytes = api.download("/api/export/attendance", params=active_filters)
        st.download_button(
            label="📥 Download Attendance Report",
            data=attendance_bytes,
            file_name="attendance_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_attendance",
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend. Ensure the server is running on http://localhost:8000.")
    except Exception as exc:
        st.error(f"Failed to prepare attendance export: {exc}")

# ── Download Student List ─────────────────────────────────────────────────────
with dl_col2:
    st.markdown("**Student List**")
    st.caption(
        "Exports the full list of registered students (all departments) "
        "to an Excel file."
    )
    try:
        students_bytes = api.download("/api/export/students")
        st.download_button(
            label="📥 Download Student List",
            data=students_bytes,
            file_name="students_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_students",
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend. Ensure the server is running on http://localhost:8000.")
    except Exception as exc:
        st.error(f"Failed to prepare student export: {exc}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Summary Statistics
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Summary Statistics")
st.caption("Attendance counts aggregated from trend data — daily (last 7 days), weekly (last 4 weeks), monthly (last 6 months).")

with st.spinner("Loading trend statistics…"):
    daily_trends   = fetch_trends("daily")
    weekly_trends  = fetch_trends("weekly")
    monthly_trends = fetch_trends("monthly")

daily_count   = sum_trend_count(daily_trends)
weekly_count  = sum_trend_count(weekly_trends)
monthly_count = sum_trend_count(monthly_trends)

# ── Metric cards ──────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("📅 Daily Total (last 7 days)",    daily_count)
m2.metric("📆 Weekly Total (last 4 weeks)",  weekly_count)
m3.metric("🗓️ Monthly Total (last 6 months)", monthly_count)

st.divider()

# ── Trend detail tables ───────────────────────────────────────────────────────
trend_tab1, trend_tab2, trend_tab3 = st.tabs(["Daily", "Weekly", "Monthly"])

with trend_tab1:
    st.markdown("**Daily Attendance Counts**")
    if daily_trends:
        df_daily = pd.DataFrame(daily_trends)
        df_daily.columns = [c.replace("_", " ").title() for c in df_daily.columns]
        data_table(df_daily, key="trends_daily_table")
    else:
        st.info("No daily trend data available.")

with trend_tab2:
    st.markdown("**Weekly Attendance Counts**")
    if weekly_trends:
        df_weekly = pd.DataFrame(weekly_trends)
        df_weekly.columns = [c.replace("_", " ").title() for c in df_weekly.columns]
        data_table(df_weekly, key="trends_weekly_table")
    else:
        st.info("No weekly trend data available.")

with trend_tab3:
    st.markdown("**Monthly Attendance Counts**")
    if monthly_trends:
        df_monthly = pd.DataFrame(monthly_trends)
        df_monthly.columns = [c.replace("_", " ").title() for c in df_monthly.columns]
        data_table(df_monthly, key="trends_monthly_table")
    else:
        st.info("No monthly trend data available.")
