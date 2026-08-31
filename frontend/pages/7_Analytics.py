"""
Analytics page — AI Attendance Monitoring System

Tabs:
  • Overview      — 30-day line chart, department bar chart, present/absent pie chart
  • Student Analysis — student selector, metric cards, attendance heatmap

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from components.metric_card import metric_card  # noqa: E402
from utils.api_client import APIClient  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics — Attendance Monitor",
    page_icon="📈",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe API GET
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(path: str, params: dict | None = None) -> dict | list | None:
    """Perform a GET request and return parsed JSON, or None on any error."""
    try:
        resp = api.get(path, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the backend. "
            "Ensure the server is running on http://localhost:8000."
        )
        return None
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error {exc.response.status_code}: {detail}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Analytics")
st.caption("Visualise attendance trends, department breakdowns, and individual student stats.")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_student = st.tabs(["📊 Overview", "🎓 Student Analysis"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Row 1: 30-day line chart ──────────────────────────────────────────────
    st.subheader("📅 30-Day Attendance Trend")

    with st.spinner("Loading trend data…"):
        trend_raw = safe_get("/api/analytics/trends", params={"period": "daily"})

    if trend_raw is not None:
        # API may return list or dict with a data/records key
        if isinstance(trend_raw, list):
            trend_records = trend_raw
        else:
            trend_records = trend_raw.get("data", trend_raw.get("records", []))

        # Take the last 30 entries
        trend_records = trend_records[-30:] if len(trend_records) > 30 else trend_records

        if trend_records:
            df_trend = pd.DataFrame(trend_records)
            date_col  = next(
                (c for c in df_trend.columns if "date" in c.lower()),
                df_trend.columns[0],
            )
            count_col = next(
                (c for c in df_trend.columns if "count" in c.lower()),
                df_trend.columns[-1],
            )

            fig_line = px.line(
                df_trend,
                x=date_col,
                y=count_col,
                markers=True,
                labels={date_col: "Date", count_col: "Students Present"},
                color_discrete_sequence=["#2980b9"],
            )
            fig_line.update_traces(
                hovertemplate="%{x}<br>Present: %{y}<extra></extra>",
                line=dict(width=2),
                marker=dict(size=6),
            )
            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Students Present",
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No daily trend data available for the past 30 days.")
    else:
        st.info("Trend data could not be loaded.")

    st.divider()

    # ── Row 2: Department bar chart + Pie chart ───────────────────────────────
    col_bar, col_pie = st.columns([3, 2])

    # ── Department bar chart ──────────────────────────────────────────────────
    with col_bar:
        st.subheader("🏢 Department-Wise Attendance")

        with st.spinner("Loading department data…"):
            dept_raw = safe_get("/api/analytics/department")

        if dept_raw is not None:
            if isinstance(dept_raw, list):
                dept_records = dept_raw
            else:
                dept_records = dept_raw.get("data", dept_raw.get("departments", []))

            if dept_records:
                df_dept = pd.DataFrame(dept_records)

                # Normalise column names
                dept_col    = next(
                    (c for c in df_dept.columns if "dept" in c.lower()),
                    df_dept.columns[0],
                )
                present_col = next(
                    (c for c in df_dept.columns if "present" in c.lower()),
                    None,
                )
                total_col   = next(
                    (c for c in df_dept.columns if "total" in c.lower()),
                    None,
                )

                if present_col and total_col:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            name="Present",
                            x=df_dept[dept_col].astype(str),
                            y=df_dept[present_col],
                            marker_color="#2ecc71",
                            hovertemplate="%{x}<br>Present: %{y}<extra></extra>",
                        )
                    )
                    # Absent = total - present
                    absent_vals = df_dept[total_col] - df_dept[present_col]
                    fig_bar.add_trace(
                        go.Bar(
                            name="Absent",
                            x=df_dept[dept_col].astype(str),
                            y=absent_vals,
                            marker_color="#e74c3c",
                            hovertemplate="%{x}<br>Absent: %{y}<extra></extra>",
                        )
                    )
                    fig_bar.update_layout(
                        barmode="stack",
                        xaxis_title="Department",
                        yaxis_title="Students",
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    # Fallback: just show whatever columns exist
                    fig_bar = px.bar(
                        df_dept,
                        x=dept_col,
                        y=df_dept.columns[-1],
                        color_discrete_sequence=["#2ecc71"],
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No department data available.")
        else:
            st.info("Department data could not be loaded.")

    # ── Pie chart: present vs absent today ────────────────────────────────────
    with col_pie:
        st.subheader("🥧 Today's Attendance Split")

        with st.spinner("Loading dashboard stats…"):
            dash_raw = safe_get("/api/analytics/dashboard")

        if dash_raw is not None:
            present_today = dash_raw.get("present_today", 0)
            absent_today  = dash_raw.get("absent_today", 0)

            if present_today > 0 or absent_today > 0:
                fig_pie = go.Figure(
                    go.Pie(
                        labels=["Present", "Absent"],
                        values=[present_today, absent_today],
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=0.35,
                        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                    )
                )
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=350,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No attendance data recorded yet for today.")
        else:
            st.info("Dashboard stats could not be loaded.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Student Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab_student:

    st.subheader("🎓 Individual Student Analysis")

    # ── Fetch student list ────────────────────────────────────────────────────
    with st.spinner("Loading student list…"):
        students_raw = safe_get("/api/students")

    students: list[dict] = []
    if students_raw is not None:
        if isinstance(students_raw, list):
            students = students_raw
        else:
            students = students_raw.get("students", students_raw.get("data", []))

    if not students:
        st.info(
            "No students found. "
            "Please register students in the Student Management page first."
        )
        st.stop()

    # Build label → roll_number map for the selectbox
    student_options = {
        f"{s.get('name', 'Unknown')} ({s.get('roll_number', '')})": s.get("roll_number", "")
        for s in students
    }

    selected_label = st.selectbox(
        "Select a student",
        options=list(student_options.keys()),
        index=0,
        key="analytics_student_selector",
        help="Choose a student to view their attendance statistics and heatmap.",
    )

    selected_rn: str = student_options.get(selected_label, "")

    if not selected_rn:
        st.warning("Could not determine the roll number for the selected student.")
        st.stop()

    st.divider()

    # ── Fetch student analytics ───────────────────────────────────────────────
    with st.spinner(f"Loading analytics for {selected_label}…"):
        student_stats = safe_get(f"/api/analytics/student/{selected_rn}")
        heatmap_raw   = safe_get(
            "/api/analytics/heatmap",
            params={"roll_number": selected_rn},
        )

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("📊 Attendance Summary")

    if student_stats is not None:
        total_days     = student_stats.get("total_days", student_stats.get("total_present", 0) + student_stats.get("total_absent", 0))
        total_present  = student_stats.get("total_present", student_stats.get("present_count", 0))
        total_absent   = student_stats.get("total_absent",  student_stats.get("absent_count",  0))
        attendance_pct = student_stats.get(
            "attendance_percentage",
            student_stats.get("percentage", 0.0),
        )

        # Recompute percentage if not provided but counts are
        if attendance_pct == 0.0 and total_days > 0:
            attendance_pct = round((total_present / total_days) * 100, 2)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card(label="Total Days",     value=total_days,     icon="📅")
        with m2:
            metric_card(label="Total Present",  value=total_present,  icon="✅")
        with m3:
            metric_card(label="Total Absent",   value=total_absent,   icon="❌")
        with m4:
            metric_card(
                label="Attendance %",
                value=f"{attendance_pct:.1f}%",
                icon="📈",
            )
    else:
        st.info("Student statistics could not be loaded.")

    st.divider()

    # ── Attendance heatmap ────────────────────────────────────────────────────
    st.subheader("🗓️ Attendance Heatmap")

    if heatmap_raw is not None:
        # heatmap_raw is a dict of {date_str: "Present"/"Absent"}
        if isinstance(heatmap_raw, dict):
            heatmap_dict: dict[str, str] = heatmap_raw
        else:
            # May be wrapped in a data key
            heatmap_dict = heatmap_raw.get("data", {}) if isinstance(heatmap_raw, dict) else {}

        if heatmap_dict:
            # Build a DataFrame: one row per date
            df_heatmap = pd.DataFrame(
                [
                    {
                        "date":   date_str,
                        "status": status_str,
                        "value":  1 if status_str == "Present" else 0,
                    }
                    for date_str, status_str in heatmap_dict.items()
                ]
            )
            df_heatmap["date"] = pd.to_datetime(df_heatmap["date"])
            df_heatmap = df_heatmap.sort_values("date")

            # Add week and day-of-week columns for a calendar-like layout
            df_heatmap["day_of_week"] = df_heatmap["date"].dt.day_name()
            df_heatmap["week"]        = df_heatmap["date"].dt.strftime("Week %U")
            df_heatmap["date_str"]    = df_heatmap["date"].dt.strftime("%Y-%m-%d")

            # Pivot to week × day-of-week matrix
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            df_pivot = (
                df_heatmap
                .pivot_table(
                    index="day_of_week",
                    columns="date_str",
                    values="value",
                    aggfunc="first",
                )
                .reindex(dow_order)
            )

            # Hovertext matrix (same shape as pivot)
            hover_pivot = (
                df_heatmap
                .pivot_table(
                    index="day_of_week",
                    columns="date_str",
                    values="status",
                    aggfunc="first",
                )
                .reindex(dow_order)
            )

            fig_heatmap = go.Figure(
                go.Heatmap(
                    z=df_pivot.values.tolist(),
                    x=df_pivot.columns.tolist(),
                    y=df_pivot.index.tolist(),
                    text=hover_pivot.values.tolist(),
                    hovertemplate="Date: %{x}<br>Day: %{y}<br>Status: %{text}<extra></extra>",
                    colorscale=[
                        [0.0, "#e74c3c"],   # Absent → red
                        [1.0, "#2ecc71"],   # Present → green
                    ],
                    showscale=True,
                    colorbar=dict(
                        tickvals=[0, 1],
                        ticktext=["Absent", "Present"],
                        thickness=15,
                    ),
                    zmin=0,
                    zmax=1,
                )
            )
            fig_heatmap.update_layout(
                xaxis_title="Date",
                yaxis_title="Day of Week",
                margin=dict(l=20, r=20, t=40, b=60),
                height=320,
                xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        else:
            st.info("No attendance data available to build a heatmap for this student.")
    else:
        st.info("Heatmap data could not be loaded.")
