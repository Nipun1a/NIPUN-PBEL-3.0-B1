"""
Live Attendance page — AI Attendance Monitoring System

Workflow:
  1. Fetch recognition settings (threshold, cooldown) from /api/settings once at load.
  2. User clicks "Start Camera" to activate the webcam widget.
  3. Each captured frame is base64-encoded and POSTed to /api/recognition/process-frame.
  4. The annotated frame is decoded and displayed with st.image.
  5. Every recognition result is appended to st.session_state.log and shown as a dataframe.
  6. Known attendances show a green st.success; unknown faces show a yellow st.warning.
  7. "Stop Camera" hides the camera widget.
"""

import base64
import sys
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# Make sure the components/utils packages are importable when running from the
# frontend/ directory (Streamlit adds the script dir to sys.path automatically,
# but be defensive in case the CWD differs).
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PAGES_PARENT = os.path.dirname(_FRONTEND_DIR)  # frontend/
for _p in (_FRONTEND_DIR, _PAGES_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.webcam_capture import capture_and_process  # noqa: E402
from utils.api_client import APIClient  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Attendance — Attendance Monitor",
    page_icon="📷",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Session state defaults ────────────────────────────────────────────────────
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "log" not in st.session_state:
    st.session_state.log = []  # list of dicts: timestamp, name, roll_number, …

# ── Page title ────────────────────────────────────────────────────────────────
st.title("📷 Live Attendance")
st.caption(
    "Capture frames from your webcam to mark attendance automatically. "
    "Each captured photo is sent to the recognition pipeline."
)

# ── Fetch settings once at page load ─────────────────────────────────────────
settings = {}

try:
    resp = api.get("/api/settings")
    resp.raise_for_status()
    settings = resp.json()
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
    st.error(f"Unexpected error while loading settings: {exc}")
    st.stop()

recognition_threshold = settings.get("recognition_threshold", 0.6)
cooldown_period       = settings.get("cooldown_period_seconds", 300)

# ── Settings info strip ───────────────────────────────────────────────────────
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.info(f"🎯 Recognition threshold: **{recognition_threshold}**")
with info_col2:
    st.info(f"⏱️ Cooldown period: **{cooldown_period} seconds**")

st.divider()

# ── Camera toggle controls ────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])

with ctrl_col1:
    if st.button(
        "▶ Start Camera",
        disabled=st.session_state.camera_active,
        use_container_width=True,
    ):
        st.session_state.camera_active = True
        st.rerun()

with ctrl_col2:
    if st.button(
        "⏹ Stop Camera",
        disabled=not st.session_state.camera_active,
        use_container_width=True,
    ):
        st.session_state.camera_active = False
        st.rerun()

# ── Main layout: camera | annotated frame ────────────────────────────────────
cam_col, frame_col = st.columns([1, 1])

notification_placeholder = st.empty()

if st.session_state.camera_active:
    with cam_col:
        st.subheader("📸 Capture Frame")
        st.caption("Click the camera button below to capture and process a frame.")

        # capture_and_process uses st.camera_input internally, encodes to JPEG
        # base64, POSTs to the endpoint, and returns the parsed JSON (or None).
        try:
            result = capture_and_process(api, "/api/recognition/process-frame")
        except requests.exceptions.ConnectionError:
            st.error(
                "Lost connection to the backend. "
                "Ensure the server is running on http://localhost:8000."
            )
            result = None
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Error {exc.response.status_code}: {detail}")
            result = None
        except Exception as exc:
            st.error(f"Unexpected error during frame processing: {exc}")
            result = None

    # ── Process API response ──────────────────────────────────────────────────
    if result is not None:
        # Decode and display the annotated frame
        annotated_b64 = result.get("annotated_frame", "")
        if annotated_b64:
            with frame_col:
                st.subheader("🖼️ Annotated Frame")
                try:
                    annotated_bytes = base64.b64decode(annotated_b64)
                    st.image(annotated_bytes, channels="BGR", use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not decode annotated frame: {exc}")

        # ── Update log and show notifications ────────────────────────────────
        results_list = result.get("results", [])
        now_str = datetime.now().strftime("%H:%M:%S")

        for recognition_result in results_list:
            # Append to session log
            st.session_state.log.append(
                {
                    "timestamp":        now_str,
                    "name":             recognition_result.get("name", ""),
                    "roll_number":      recognition_result.get("roll_number", ""),
                    "confidence_score": recognition_result.get("confidence_score", 0.0),
                    "attendance_marked": recognition_result.get("attendance_marked", False),
                }
            )

            # Notifications
            name   = recognition_result.get("name", "Unknown")
            roll   = recognition_result.get("roll_number", "")
            status = recognition_result.get("recognition_status", "")
            marked = recognition_result.get("attendance_marked", False)

            if status == "Unknown":
                with notification_placeholder.container():
                    st.warning(
                        f"⚠️ Unknown face detected (confidence: "
                        f"{recognition_result.get('confidence_score', 0.0):.2f})"
                    )
            elif marked:
                with notification_placeholder.container():
                    st.success(
                        f"✅ Attendance marked — **{name}** "
                        f"(Roll: {roll}, "
                        f"confidence: {recognition_result.get('confidence_score', 0.0):.2f})"
                    )
            elif recognition_result.get("duplicate", False):
                with notification_placeholder.container():
                    st.info(
                        f"ℹ️ Duplicate — **{name}** (Roll: {roll}) already marked "
                        "within the cooldown period."
                    )

else:
    # Camera is stopped — show placeholder in both columns
    with cam_col:
        st.subheader("📸 Capture Frame")
        st.info("Camera is stopped. Click **▶ Start Camera** to begin.")
    with frame_col:
        st.subheader("🖼️ Annotated Frame")
        st.info("No frame captured yet.")

st.divider()

# ── Attendance log ────────────────────────────────────────────────────────────
log_header_col, log_clear_col = st.columns([5, 1])

with log_header_col:
    st.subheader("📋 Recognition Log")

with log_clear_col:
    if st.button("🗑️ Clear Log", use_container_width=True):
        st.session_state.log = []
        st.rerun()

if st.session_state.log:
    df_log = pd.DataFrame(
        st.session_state.log,
        columns=[
            "timestamp",
            "name",
            "roll_number",
            "confidence_score",
            "attendance_marked",
        ],
    )

    # Rename columns to friendlier headers for display
    df_log.rename(
        columns={
            "timestamp":        "Timestamp",
            "name":             "Name",
            "roll_number":      "Roll Number",
            "confidence_score": "Confidence",
            "attendance_marked": "Marked",
        },
        inplace=True,
    )

    # Format confidence to 3 decimal places
    df_log["Confidence"] = df_log["Confidence"].map(lambda x: round(float(x), 3))

    st.dataframe(
        df_log,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Total entries: {len(st.session_state.log)}")
else:
    st.info("No recognition events recorded yet. Start the camera and capture frames to populate this log.")
