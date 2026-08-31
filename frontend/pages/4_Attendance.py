"""
Attendance Management page — AI Attendance Monitoring System

Three-tab layout:
  • Today's Attendance  — load /api/attendance/today, show table + summary metrics
  • Filter & Search     — date/roll/name/status filters → /api/attendance, inline Edit & Delete
  • Add Manual Entry    — form → POST /api/attendance/manual

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

import os
import sys
from datetime import date, datetime

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
    page_title="Attendance — Attendance Monitor",
    page_icon="📅",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Column display configuration ─────────────────────────────────────────────
_RENAME_MAP = {
    "roll_number":      "Roll Number",
    "name":             "Name",
    "department":       "Department",
    "date":             "Date",
    "time":             "Time",
    "confidence_score": "Confidence Score",
    "status":           "Status",
    "marked_by":        "Marked By",
    "id":               "ID",
}

_DISPLAY_COLS = [
    "Roll Number", "Name", "Department", "Date",
    "Time", "Confidence Score", "Status",
]

_STATUS_OPTIONS = ["Present", "Absent", "Late"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert raw API records to a display-ready DataFrame."""
    if not records:
        return pd.DataFrame(columns=_DISPLAY_COLS)
    df = pd.DataFrame(records)
    df = df.rename(columns={k: v for k, v in _RENAME_MAP.items() if k in df.columns})
    keep = [c for c in _DISPLAY_COLS if c in df.columns]
    return df[keep]


def compute_summary(records: list[dict]) -> tuple[int, int, float]:
    """Return (present_count, absent_count, attendance_pct) from raw records."""
    total = len(records)
    present = sum(1 for r in records if r.get("status", "") == "Present")
    absent = total - present
    pct = round((present / total) * 100, 2) if total > 0 else 0.0
    return present, absent, pct


def show_summary(records: list[dict]) -> None:
    """Render three st.metric cards summarising the records."""
    present, absent, pct = compute_summary(records)
    total = len(records)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", total)
    c2.metric("✅ Present", present)
    c3.metric("❌ Absent", absent)
    c4.metric("📊 Attendance %", f"{pct}%")


def fetch_today() -> list[dict]:
    """Fetch today's attendance from /api/attendance/today."""
    try:
        resp = api.get("/api/attendance/today")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("records", [])
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
        st.error(f"Unexpected error: {exc}")
        return []


def fetch_filtered(
    filter_date: str | None,
    roll_number: str,
    name: str,
    status: str,
) -> list[dict]:
    """Fetch attendance records with optional filters from /api/attendance."""
    params: dict = {}
    if filter_date:
        params["date"] = filter_date
    if roll_number.strip():
        params["roll_number"] = roll_number.strip()
    if name.strip():
        params["name"] = name.strip()
    if status and status != "All":
        params["status"] = status

    try:
        resp = api.get("/api/attendance", params=params)
        resp.raise_for_status()
        data = resp.json()
        # API returns paginated envelope or plain list
        if isinstance(data, list):
            return data
        return data.get("records", [])
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
        st.error(f"Unexpected error: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.title("📅 Attendance Management")
st.caption("View, filter, edit, delete, and manually add attendance records.")

tab_today, tab_filter, tab_manual = st.tabs(
    ["📋 Today's Attendance", "🔍 Filter & Search", "✏️ Add Manual Entry"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Today's Attendance
# ═════════════════════════════════════════════════════════════════════════════
with tab_today:
    today_str = date.today().strftime("%Y-%m-%d")
    st.subheader(f"Today's Attendance — {today_str}")

    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        if st.button("🔄 Refresh", key="refresh_today", use_container_width=True):
            st.rerun()

    with st.spinner("Fetching today's attendance…"):
        today_records = fetch_today()

    if today_records:
        show_summary(today_records)
        st.divider()

    df_today = records_to_df(today_records)
    data_table(df_today, key="today_table")

    if today_records:
        st.divider()
        st.subheader("Row Actions")
        st.caption("Expand a record below to edit or delete it.")

        for rec in today_records:
            rec_id   = rec.get("id")
            rec_rn   = rec.get("roll_number", "")
            rec_name = rec.get("name", "")
            rec_time = rec.get("time", "")
            rec_status = rec.get("status", "Present")

            label = f"📌 {rec_name}  (Roll: {rec_rn})  —  {rec_time}  —  {rec_status}"
            with st.expander(label):
                edit_col, del_col = st.columns(2)

                # ── EDIT ──────────────────────────────────────────────────────
                with edit_col:
                    st.markdown("**Edit Record**")
                    with st.form(key=f"edit_today_{rec_id}"):
                        new_status = st.selectbox(
                            "Status",
                            options=_STATUS_OPTIONS,
                            index=_STATUS_OPTIONS.index(rec_status) if rec_status in _STATUS_OPTIONS else 0,
                            key=f"edit_today_status_{rec_id}",
                        )
                        new_time = st.text_input(
                            "Time (HH:MM:SS)",
                            value=rec_time,
                            key=f"edit_today_time_{rec_id}",
                        )
                        save_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)

                    if save_edit:
                        try:
                            resp = api.put(
                                f"/api/attendance/{rec_id}",
                                json={"status": new_status, "time": new_time},
                            )
                            resp.raise_for_status()
                            st.success("✅ Record updated successfully.")
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend.")
                        except requests.exceptions.HTTPError as exc:
                            try:
                                detail = exc.response.json().get("detail", str(exc))
                            except Exception:
                                detail = str(exc)
                            st.error(f"Error {exc.response.status_code}: {detail}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")

                # ── DELETE ────────────────────────────────────────────────────
                with del_col:
                    st.markdown("**Delete Record**")
                    st.warning(
                        f"Delete attendance record for **{rec_name}** (Roll: {rec_rn}) "
                        f"at **{rec_time}**? This action is irreversible."
                    )
                    confirm_del = st.checkbox(
                        "Confirm deletion",
                        key=f"confirm_today_del_{rec_id}",
                    )
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_today_btn_{rec_id}",
                        disabled=not confirm_del,
                        use_container_width=True,
                    ):
                        try:
                            resp = api.delete(f"/api/attendance/{rec_id}")
                            resp.raise_for_status()
                            st.success("✅ Record deleted successfully.")
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend.")
                        except requests.exceptions.HTTPError as exc:
                            try:
                                detail = exc.response.json().get("detail", str(exc))
                            except Exception:
                                detail = str(exc)
                            st.error(f"Error {exc.response.status_code}: {detail}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Filter & Search
# ═════════════════════════════════════════════════════════════════════════════
with tab_filter:
    st.subheader("Filter & Search Attendance Records")

    # ── Filter controls ───────────────────────────────────────────────────────
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 2])

    with f_col1:
        use_date = st.checkbox("Filter by date", key="filter_use_date")
        filter_date_val: str | None = None
        if use_date:
            picked = st.date_input("Date", value=date.today(), key="filter_date")
            filter_date_val = picked.strftime("%Y-%m-%d") if picked else None

    with f_col2:
        filter_roll = st.text_input(
            "Roll Number",
            placeholder="e.g. 101",
            key="filter_roll",
        )

    with f_col3:
        filter_name = st.text_input(
            "Name",
            placeholder="e.g. Nipun",
            key="filter_name",
        )

    with f_col4:
        filter_status = st.selectbox(
            "Status",
            options=["All"] + _STATUS_OPTIONS,
            key="filter_status",
        )

    refresh_f_col, _ = st.columns([1, 5])
    with refresh_f_col:
        if st.button("🔍 Search", key="do_filter_search", use_container_width=True):
            st.rerun()  # filters are already in session_state; rerun re-fetches

    st.divider()

    # ── Fetch and display ─────────────────────────────────────────────────────
    with st.spinner("Loading attendance records…"):
        filtered_records = fetch_filtered(
            filter_date_val,
            st.session_state.get("filter_roll", ""),
            st.session_state.get("filter_name", ""),
            st.session_state.get("filter_status", "All"),
        )

    if filtered_records:
        show_summary(filtered_records)
        st.caption(f"Showing {len(filtered_records)} record(s)")
        st.divider()

    df_filtered = records_to_df(filtered_records)
    data_table(df_filtered, key="filter_table")

    # ── Per-record edit / delete expanders ────────────────────────────────────
    if filtered_records:
        st.divider()
        st.subheader("Row Actions")
        st.caption("Expand a record below to edit or delete it.")

        for rec in filtered_records:
            rec_id     = rec.get("id")
            rec_rn     = rec.get("roll_number", "")
            rec_name   = rec.get("name", "")
            rec_date   = rec.get("date", "")
            rec_time   = rec.get("time", "")
            rec_status = rec.get("status", "Present")

            label = f"📌 {rec_name}  (Roll: {rec_rn})  —  {rec_date}  {rec_time}  —  {rec_status}"
            with st.expander(label):
                edit_col, del_col = st.columns(2)

                # ── EDIT ──────────────────────────────────────────────────────
                with edit_col:
                    st.markdown("**Edit Record**")
                    with st.form(key=f"edit_filter_{rec_id}"):
                        upd_status = st.selectbox(
                            "Status",
                            options=_STATUS_OPTIONS,
                            index=_STATUS_OPTIONS.index(rec_status) if rec_status in _STATUS_OPTIONS else 0,
                            key=f"edit_filter_status_{rec_id}",
                        )
                        upd_time = st.text_input(
                            "Time (HH:MM:SS)",
                            value=rec_time,
                            key=f"edit_filter_time_{rec_id}",
                        )
                        save_upd = st.form_submit_button("💾 Save Changes", use_container_width=True)

                    if save_upd:
                        try:
                            resp = api.put(
                                f"/api/attendance/{rec_id}",
                                json={"status": upd_status, "time": upd_time},
                            )
                            resp.raise_for_status()
                            st.success("✅ Record updated successfully.")
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend.")
                        except requests.exceptions.HTTPError as exc:
                            try:
                                detail = exc.response.json().get("detail", str(exc))
                            except Exception:
                                detail = str(exc)
                            st.error(f"Error {exc.response.status_code}: {detail}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")

                # ── DELETE ────────────────────────────────────────────────────
                with del_col:
                    st.markdown("**Delete Record**")
                    st.warning(
                        f"Delete attendance record for **{rec_name}** (Roll: {rec_rn}) "
                        f"on **{rec_date}** at **{rec_time}**? This is irreversible."
                    )
                    confirm_del_f = st.checkbox(
                        "Confirm deletion",
                        key=f"confirm_filter_del_{rec_id}",
                    )
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_filter_btn_{rec_id}",
                        disabled=not confirm_del_f,
                        use_container_width=True,
                    ):
                        try:
                            resp = api.delete(f"/api/attendance/{rec_id}")
                            resp.raise_for_status()
                            st.success("✅ Record deleted successfully.")
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend.")
                        except requests.exceptions.HTTPError as exc:
                            try:
                                detail = exc.response.json().get("detail", str(exc))
                            except Exception:
                                detail = str(exc)
                            st.error(f"Error {exc.response.status_code}: {detail}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Add Manual Entry
# ═════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Add Manual Attendance Entry")
    st.caption(
        "Use this form to manually record attendance for a student. "
        "The entry will be stored with **marked_by = manual**."
    )

    with st.form("manual_attendance_form", clear_on_submit=True):
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            manual_roll = st.text_input(
                "Roll Number *",
                placeholder="e.g. 101",
                help="Must match an existing student roll number",
            )
            manual_date = st.date_input(
                "Date *",
                value=date.today(),
                help="Date of attendance",
            )

        with m_col2:
            manual_time = st.time_input(
                "Time *",
                value=datetime.now().time(),
                help="Time of attendance",
            )
            manual_status = st.selectbox(
                "Status *",
                options=_STATUS_OPTIONS,
                index=0,
            )

        submit_manual = st.form_submit_button(
            "✅ Add Entry",
            use_container_width=True,
            type="primary",
        )

    if submit_manual:
        errors: list[str] = []
        if not manual_roll.strip():
            errors.append("Roll Number is required.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            payload = {
                "roll_number": manual_roll.strip(),
                "date":        manual_date.strftime("%Y-%m-%d"),
                "time":        manual_time.strftime("%H:%M:%S"),
                "status":      manual_status,
            }
            try:
                resp = api.post("/api/attendance/manual", json=payload)
                resp.raise_for_status()
                st.success(
                    f"✅ Manual attendance entry added for roll number "
                    f"**{manual_roll.strip()}** on {payload['date']} at {payload['time']}."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend. "
                    "Ensure the server is running on http://localhost:8000."
                )
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    st.error(
                        f"⚠️ No student found with roll number **{manual_roll.strip()}**. "
                        "Please register the student first."
                    )
                else:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Error {exc.response.status_code}: {detail}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    st.divider()
    st.caption(
        "Manual entries bypass the face recognition pipeline and are flagged as "
        "**manual** in the `Marked By` field. They appear in all reports and exports."
    )
