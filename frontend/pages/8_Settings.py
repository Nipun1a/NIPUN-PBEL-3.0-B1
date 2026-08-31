"""
Settings page — AI Attendance Monitoring System

Allows the Admin to view and update all 6 system configuration keys:
  - recognition_threshold   (float 0.0–1.0)
  - cooldown_period_seconds (int   0–86400)
  - stable_frame_count      (int   1–30)
  - camera_index            (int   0–9)
  - blur_threshold          (float 0.0–500.0)
  - min_face_size           (int   10–500)

Also provides a "Reload Embeddings" button to hot-reload the face embedding
store without restarting the server.

Requirements: 17.1, 17.2, 17.3, 17.4, 17.5
"""
from __future__ import annotations

import os
import sys

import requests
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from utils.api_client import APIClient  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Settings — Attendance Monitor",
    page_icon="⚙️",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Page title ────────────────────────────────────────────────────────────────
st.title("⚙️ Settings")
st.caption(
    "Configure recognition thresholds, cooldown window, camera selection, and "
    "image quality parameters. Changes take effect immediately without a server restart."
)

st.divider()

# ── Fetch current settings ────────────────────────────────────────────────────
current_settings: dict = {}

with st.spinner("Loading current settings…"):
    try:
        resp = api.get("/api/settings")
        resp.raise_for_status()
        current_settings = resp.json()
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
        st.error(f"Unexpected error loading settings: {exc}")
        st.stop()

# Store loaded values in session state for change detection
if "loaded_settings" not in st.session_state:
    st.session_state.loaded_settings = {}
st.session_state.loaded_settings = dict(current_settings)

# ── Settings form ─────────────────────────────────────────────────────────────
st.subheader("🔧 Recognition & Camera Parameters")
st.caption("Adjust the sliders and inputs below, then click **Save Settings** to apply.")

with st.form("settings_form"):
    col1, col2 = st.columns(2)

    with col1:
        # ── Recognition Threshold ─────────────────────────────────────────────
        new_threshold = st.slider(
            "Recognition Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(current_settings.get("recognition_threshold", 0.6)),
            step=0.01,
            help="Minimum cosine-similarity score to classify a face as Known. "
                 "Valid range: 0.0 – 1.0",
        )
        st.caption("Range: 0.0 – 1.0 · Default: 0.6")

        # ── Stable Frame Count ────────────────────────────────────────────────
        new_stable_frame = st.slider(
            "Stable Frame Count",
            min_value=1,
            max_value=30,
            value=int(current_settings.get("stable_frame_count", 4)),
            step=1,
            help="Number of consecutive frames a recognition must be stable before "
                 "triggering attendance. Valid range: 1 – 30",
        )
        st.caption("Range: 1 – 30 · Default: 4")

        # ── Blur Threshold ────────────────────────────────────────────────────
        new_blur = st.slider(
            "Blur Threshold",
            min_value=0.0,
            max_value=500.0,
            value=float(current_settings.get("blur_threshold", 50.0)),
            step=1.0,
            help="Variance-of-Laplacian threshold below which a frame is considered "
                 "too blurry to process. Valid range: 0.0 – 500.0",
        )
        st.caption("Range: 0.0 – 500.0 · Default: 50.0")

    with col2:
        # ── Cooldown Period ───────────────────────────────────────────────────
        new_cooldown = st.number_input(
            "Cooldown Period (seconds)",
            min_value=0,
            max_value=86400,
            value=int(current_settings.get("cooldown_period_seconds", 300)),
            step=30,
            help="Time window in seconds during which a duplicate attendance record "
                 "for the same student is suppressed. Valid range: 0 – 86400",
        )
        st.caption("Range: 0 – 86400 · Default: 300 (5 minutes)")

        # ── Camera Index ──────────────────────────────────────────────────────
        new_camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=9,
            value=int(current_settings.get("camera_index", 0)),
            step=1,
            help="OpenCV camera device index used for the live stream. "
                 "0 = default webcam. Valid range: 0 – 9",
        )
        st.caption("Range: 0 – 9 · Default: 0")

        # ── Minimum Face Size ─────────────────────────────────────────────────
        new_min_face = st.number_input(
            "Minimum Face Size (pixels)",
            min_value=10,
            max_value=500,
            value=int(current_settings.get("min_face_size", 60)),
            step=10,
            help="Minimum bounding-box dimension (in pixels) for a detected face "
                 "to be processed. Valid range: 10 – 500",
        )
        st.caption("Range: 10 – 500 · Default: 60")

    st.divider()
    save_clicked = st.form_submit_button(
        "💾 Save Settings",
        use_container_width=True,
        type="primary",
    )

# ── Handle save ───────────────────────────────────────────────────────────────
if save_clicked:
    loaded = st.session_state.loaded_settings

    # Build payload with only the changed values
    payload: dict = {}
    if abs(new_threshold - float(loaded.get("recognition_threshold", 0.6))) > 1e-9:
        payload["recognition_threshold"] = new_threshold
    if new_cooldown != int(loaded.get("cooldown_period_seconds", 300)):
        payload["cooldown_period_seconds"] = new_cooldown
    if new_stable_frame != int(loaded.get("stable_frame_count", 4)):
        payload["stable_frame_count"] = new_stable_frame
    if new_camera_index != int(loaded.get("camera_index", 0)):
        payload["camera_index"] = new_camera_index
    if abs(new_blur - float(loaded.get("blur_threshold", 50.0))) > 1e-9:
        payload["blur_threshold"] = new_blur
    if new_min_face != int(loaded.get("min_face_size", 60)):
        payload["min_face_size"] = new_min_face

    if not payload:
        st.info("ℹ️ No changes detected. Settings are already up to date.")
    else:
        # Camera index change notice (Req 17.4)
        if "camera_index" in payload:
            st.info(
                f"📷 **Camera index changed to {new_camera_index}.** "
                "The live camera stream will use this new index on the next session start. "
                "Refresh the Live Attendance page to apply."
            )

        with st.spinner("Saving settings…"):
            try:
                put_resp = api.put("/api/settings", json=payload)

                if put_resp.status_code == 200:
                    updated = put_resp.json()
                    st.success(
                        f"✅ Settings saved successfully. "
                        f"Updated keys: **{', '.join(payload.keys())}**."
                    )
                    # Refresh loaded settings so next diff is correct
                    st.session_state.loaded_settings = dict(updated)
                    st.rerun()

                elif put_resp.status_code == 422:
                    # Field-level validation errors from Pydantic (Req 17.2)
                    try:
                        errors = put_resp.json()
                        detail = errors.get("detail", errors)
                    except Exception:
                        detail = put_resp.text

                    st.error("⚠️ Validation error — one or more values are out of range:")
                    if isinstance(detail, list):
                        for err in detail:
                            loc   = " → ".join(str(x) for x in err.get("loc", []))
                            msg   = err.get("msg", str(err))
                            st.error(f"  • **{loc}**: {msg}")
                    else:
                        st.error(str(detail))

                else:
                    put_resp.raise_for_status()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend. "
                    "Ensure the server is running on http://localhost:8000."
                )
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"Error {exc.response.status_code}: {detail}")
            except Exception as exc:
                st.error(f"Unexpected error saving settings: {exc}")

st.divider()

# ── Reload Embeddings ─────────────────────────────────────────────────────────
st.subheader("🔄 Reload Face Embeddings")
st.caption(
    "Hot-reload the face embedding store from disk. Use this after generating "
    "new embeddings for a student if the recogniser hasn't picked them up yet. "
    "No server restart is required."
)

if st.button(
    "⚡ Reload Embeddings",
    type="secondary",
    key="reload_embeddings_btn",
):
    with st.spinner("Reloading embeddings…"):
        try:
            reload_resp = api.post("/api/recognition/reload-embeddings")
            reload_resp.raise_for_status()
            reload_data = reload_resp.json()
            student_count = reload_data.get("student_count", "?")
            message       = reload_data.get("message", "Embeddings reloaded successfully.")
            st.success(
                f"✅ {message} "
                f"**{student_count}** student embedding(s) now loaded in memory."
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. "
                "Ensure the server is running on http://localhost:8000."
            )
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Error {exc.response.status_code}: {detail}")
        except Exception as exc:
            st.error(f"Unexpected error reloading embeddings: {exc}")

st.divider()

# ── Current Settings Reference ────────────────────────────────────────────────
st.subheader("📋 Current Settings Reference")
st.caption("Last-loaded values from the backend database.")

ref_data = [
    {
        "Setting":       "Recognition Threshold",
        "Key":           "recognition_threshold",
        "Current Value": current_settings.get("recognition_threshold", "—"),
        "Valid Range":   "0.0 – 1.0",
        "Default":       "0.6",
    },
    {
        "Setting":       "Cooldown Period (s)",
        "Key":           "cooldown_period_seconds",
        "Current Value": current_settings.get("cooldown_period_seconds", "—"),
        "Valid Range":   "0 – 86400",
        "Default":       "300",
    },
    {
        "Setting":       "Stable Frame Count",
        "Key":           "stable_frame_count",
        "Current Value": current_settings.get("stable_frame_count", "—"),
        "Valid Range":   "1 – 30",
        "Default":       "4",
    },
    {
        "Setting":       "Camera Index",
        "Key":           "camera_index",
        "Current Value": current_settings.get("camera_index", "—"),
        "Valid Range":   "0 – 9",
        "Default":       "0",
    },
    {
        "Setting":       "Blur Threshold",
        "Key":           "blur_threshold",
        "Current Value": current_settings.get("blur_threshold", "—"),
        "Valid Range":   "0.0 – 500.0",
        "Default":       "50.0",
    },
    {
        "Setting":       "Minimum Face Size (px)",
        "Key":           "min_face_size",
        "Current Value": current_settings.get("min_face_size", "—"),
        "Valid Range":   "10 – 500",
        "Default":       "60",
    },
]

import pandas as pd  # noqa: E402

df_ref = pd.DataFrame(ref_data)
st.dataframe(df_ref, hide_index=True, use_container_width=True)
