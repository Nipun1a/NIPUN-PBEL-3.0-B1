"""
Dashboard page — AI Attendance Monitoring System
Displays key metrics, a 7-day attendance bar chart, and a today's
present/absent pie chart.  Auto-refreshes every 30 seconds.
"""

import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from components.metric_card import metric_card
from utils.api_client import APIClient

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Attendance Monitor",
    page_icon="📊",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Auto-refresh logic (every 30 s) ──────────────────────────────────────────
REFRESH_INTERVAL = 30  # seconds

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

now = time.time()
elapsed = now - st.session_state.last_refresh
if elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ── Page title ────────────────────────────────────────────────────────────────
st.title("📊 Dashboard")

seconds_until_refresh = max(0, int(REFRESH_INTERVAL - elapsed))
st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s · next refresh in {seconds_until_refresh}s")

# ── Skeleton placeholders ─────────────────────────────────────────────────────
metrics_placeholder = st.empty()
charts_placeholder = st.empty()

# ── Fetch dashboard stats ─────────────────────────────────────────────────────
dashboard_data = None

with st.spinner("Loading dashboard..."):
    try:
        resp = api.get("/api/analytics/dashboard")
        resp.raise_for_status()
        dashboard_data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the backend. "
            "Ensure the server is running on http://localhost:8000."
        )
        st.stop()
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error {exc.response.status_code}: {detail}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error while loading dashboard stats: {exc}")
        st.stop()

# ── Render metric cards ───────────────────────────────────────────────────────
total_students        = dashboard_data.get("total_students", 0)
present_today         = dashboard_data.get("present_today", 0)
absent_today          = dashboard_data.get("absent_today", 0)
attendance_percentage = dashboard_data.get("attendance_percentage", 0.0)
unknown_face_count    = dashboard_data.get("unknown_face_count", 0)

with metrics_placeholder.container():
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card(
            label="Total Students",
            value=total_students,
            icon="🎓",
        )
    with col2:
        metric_card(
            label="Present Today",
            value=present_today,
            icon="✅",
        )
    with col3:
        metric_card(
            label="Absent Today",
            value=absent_today,
            icon="❌",
        )
    with col4:
        metric_card(
            label="Attendance %",
            value=f"{attendance_percentage:.1f}%",
            icon="📈",
        )
    with col5:
        metric_card(
            label="Unknown Faces",
            value=unknown_face_count,
            icon="❓",
        )

st.divider()

# ── Fetch trend data for bar chart ────────────────────────────────────────────
trend_data = None

try:
    resp = api.get("/api/analytics/trends", params={"period": "daily"})
    resp.raise_for_status()
    trend_data = resp.json()
except requests.exceptions.ConnectionError:
    st.error(
        "Cannot reach the backend while loading trend data. "
        "Ensure the server is running on http://localhost:8000."
    )
except requests.exceptions.HTTPError as exc:
    try:
        detail = exc.response.json().get("detail", str(exc))
    except Exception:
        detail = str(exc)
    st.error(f"Error {exc.response.status_code}: {detail}")
except Exception as exc:
    st.error(f"Unexpected error while loading trend data: {exc}")

# ── Render charts side-by-side ────────────────────────────────────────────────
with charts_placeholder.container():
    chart_col1, chart_col2 = st.columns([3, 2])

    # -- Bar chart: past 7 days ------------------------------------------------
    with chart_col1:
        st.subheader("📅 Attendance — Past 7 Days")

        if trend_data is not None:
            # The API may return a list directly or a dict with a "data" key
            if isinstance(trend_data, list):
                records = trend_data
            else:
                records = trend_data.get("data", trend_data.get("records", []))

            # Keep only the last 7 data points
            records = records[-7:] if len(records) > 7 else records

            if records:
                df_trend = pd.DataFrame(records)
                # Normalise column names — API returns {"date": ..., "count": ...}
                date_col  = next((c for c in df_trend.columns if "date" in c.lower()), df_trend.columns[0])
                count_col = next((c for c in df_trend.columns if "count" in c.lower()), df_trend.columns[-1])

                fig_bar = go.Figure(
                    go.Bar(
                        x=df_trend[date_col].astype(str),
                        y=df_trend[count_col],
                        marker_color="steelblue",
                        hovertemplate="%{x}<br>Present: %{y}<extra></extra>",
                    )
                )
                fig_bar.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Students Present",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=350,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No trend data available for the past 7 days.")
        else:
            st.info("Trend data could not be loaded.")

    # -- Pie chart: present vs absent today ------------------------------------
    with chart_col2:
        st.subheader("🥧 Today's Attendance Split")

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
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No attendance data recorded yet for today.")
