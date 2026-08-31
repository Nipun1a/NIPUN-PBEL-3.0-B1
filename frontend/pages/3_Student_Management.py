"""
Student Management page — AI Attendance Monitoring System

Three-tab layout:
  • Student List   — searchable, paginated table; inline Edit & Delete per row
  • Register       — form to create a new student
  • Capture & Train — webcam capture loop + Generate Embeddings

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 10.7
"""

import base64
import io
import os
import sys
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── Path setup (defensive; Streamlit adds pages/ parent automatically) ────────
_FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from components.data_table import data_table  # noqa: E402
from utils.api_client import APIClient  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Management — Attendance Monitor",
    page_icon="🎓",
    layout="wide",
)

# ── Ensure API client is available ───────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Session state defaults ────────────────────────────────────────────────────
if "capture_count" not in st.session_state:
    st.session_state.capture_count = 0        # frames captured in current session
if "capture_active" not in st.session_state:
    st.session_state.capture_active = False   # whether webcam loop is running
if "capture_roll" not in st.session_state:
    st.session_state.capture_roll = ""        # roll number being captured
if "students_cache" not in st.session_state:
    st.session_state.students_cache = []      # last-fetched student list


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fetch all students (used by multiple tabs)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_students(query: str = "") -> list[dict]:
    """Return list of student dicts from the API.  Empty list on error."""
    try:
        if query.strip():
            resp = api.get("/api/students/search", params={"q": query.strip()})
        else:
            resp = api.get("/api/students")
        resp.raise_for_status()
        data = resp.json()
        # API may return a list directly or a paginated envelope
        if isinstance(data, list):
            return data
        return data.get("students", data.get("records", data.get("items", [])))
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the backend. "
            "Ensure the server is running on http://localhost:8000."
        )
        return []
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Error {exc.response.status_code}: {detail}")
        return []
    except Exception as exc:
        st.error(f"Unexpected error fetching students: {exc}")
        return []


def students_dataframe(students: list[dict]) -> pd.DataFrame:
    """Convert a list of student dicts to a display-friendly DataFrame."""
    if not students:
        return pd.DataFrame(
            columns=["Roll Number", "Name", "Department", "Email", "Phone", "Created"]
        )
    df = pd.DataFrame(students)
    rename_map = {
        "roll_number": "Roll Number",
        "name": "Name",
        "department": "Department",
        "email": "Email",
        "phone": "Phone",
        "created_at": "Created",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Keep only display columns that exist
    keep = [c for c in rename_map.values() if c in df.columns]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎓 Student Management")
st.caption("Register students, capture face images, generate embeddings, and manage the student roster.")

tab_list, tab_register, tab_capture = st.tabs(
    ["📋 Student List", "➕ Register Student", "📸 Capture & Train"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Student List
# ═════════════════════════════════════════════════════════════════════════════
with tab_list:
    st.subheader("Student Roster")

    # ── Search bar ────────────────────────────────────────────────────────────
    search_col, refresh_col = st.columns([5, 1])
    with search_col:
        search_query = st.text_input(
            "Search students",
            placeholder="Search by name or roll number…",
            key="student_search",
            label_visibility="collapsed",
        )
    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_list"):
            st.session_state.students_cache = []
            st.rerun()

    # ── Fetch students (cached until manual refresh or action) ────────────────
    with st.spinner("Loading students…"):
        students = fetch_students(search_query)
        st.session_state.students_cache = students

    if not students:
        st.info("No students found. Use the **Register Student** tab to add the first student.")
    else:
        st.caption(f"Showing {len(students)} student(s)")

        # ── Render table ──────────────────────────────────────────────────────
        df_display = students_dataframe(students)
        data_table(df_display, key="student_table")

        st.divider()
        st.subheader("Student Actions")
        st.caption("Expand a student row below to edit or delete.")

        # ── Per-student expanders ─────────────────────────────────────────────
        for student in students:
            rn   = student.get("roll_number", "")
            name = student.get("name", "")

            with st.expander(f"👤 {name}  —  Roll: {rn}"):
                action_col1, action_col2 = st.columns(2)

                # ── EDIT ──────────────────────────────────────────────────────
                with action_col1:
                    st.markdown("**Edit Student**")
                    with st.form(key=f"edit_form_{rn}"):
                        edit_name  = st.text_input("Name",       value=student.get("name", ""),       key=f"edit_name_{rn}")
                        edit_dept  = st.text_input("Department", value=student.get("department", ""), key=f"edit_dept_{rn}")
                        edit_email = st.text_input("Email",      value=student.get("email", ""),      key=f"edit_email_{rn}")
                        edit_phone = st.text_input("Phone",      value=student.get("phone", ""),      key=f"edit_phone_{rn}")
                        submitted_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)

                    if submitted_edit:
                        if not edit_name.strip():
                            st.error("Name cannot be empty.")
                        else:
                            payload = {
                                "name":       edit_name.strip(),
                                "department": edit_dept.strip(),
                                "email":      edit_email.strip(),
                                "phone":      edit_phone.strip(),
                            }
                            try:
                                resp = api.put(f"/api/students/{rn}", json=payload)
                                resp.raise_for_status()
                                st.success(f"✅ Student **{edit_name}** updated successfully.")
                                st.session_state.students_cache = []
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
                with action_col2:
                    st.markdown("**Delete Student**")
                    st.warning(
                        f"Deleting **{name}** (Roll: {rn}) will remove the student "
                        "record, all captured images, and their face embeddings. "
                        "This action is **irreversible**."
                    )
                    confirm_key = f"confirm_delete_{rn}"
                    confirmed = st.checkbox(
                        "I understand — permanently delete this student",
                        key=confirm_key,
                    )
                    if st.button(
                        "🗑️ Delete Student",
                        key=f"delete_btn_{rn}",
                        disabled=not confirmed,
                        use_container_width=True,
                    ):
                        try:
                            resp = api.delete(f"/api/students/{rn}")
                            resp.raise_for_status()
                            st.success(f"✅ Student **{name}** deleted successfully.")
                            st.session_state.students_cache = []
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
# TAB 2 — Register Student
# ═════════════════════════════════════════════════════════════════════════════
with tab_register:
    st.subheader("Register a New Student")
    st.caption("Fill in the form below and click **Register Student** to add the student to the system.")

    with st.form("register_student_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)

        with col_a:
            reg_roll   = st.text_input("Roll Number *", placeholder="e.g. 101", help="Unique alphanumeric identifier")
            reg_name   = st.text_input("Full Name *",   placeholder="e.g. Nipun Sharma")
            reg_dept   = st.text_input("Department",    placeholder="e.g. Computer Science")

        with col_b:
            reg_email  = st.text_input("Email",         placeholder="e.g. student@example.com")
            reg_phone  = st.text_input("Phone",         placeholder="e.g. +91 9876543210")
            st.markdown("")  # spacer
            st.markdown("")  # spacer

        submit_register = st.form_submit_button(
            "✅ Register Student",
            use_container_width=True,
            type="primary",
        )

    if submit_register:
        # Basic client-side validation
        errors = []
        if not reg_roll.strip():
            errors.append("Roll Number is required.")
        if not reg_name.strip():
            errors.append("Full Name is required.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            payload = {
                "roll_number": reg_roll.strip(),
                "name":        reg_name.strip(),
                "department":  reg_dept.strip(),
                "email":       reg_email.strip(),
                "phone":       reg_phone.strip(),
            }
            try:
                resp = api.post("/api/students", json=payload)

                if resp.status_code == 201:
                    st.success(
                        f"✅ Student **{reg_name.strip()}** (Roll: {reg_roll.strip()}) "
                        "registered successfully! Switch to the **Capture & Train** tab "
                        "to capture face images."
                    )
                    st.session_state.students_cache = []
                elif resp.status_code == 409:
                    st.error(
                        f"⚠️ A student with roll number **{reg_roll.strip()}** already exists. "
                        "Please use a different roll number."
                    )
                else:
                    resp.raise_for_status()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend. "
                    "Ensure the server is running on http://localhost:8000."
                )
            except requests.exceptions.HTTPError as exc:
                # 409 already handled above; handle other HTTP errors here
                if exc.response is not None and exc.response.status_code == 409:
                    st.error(
                        f"⚠️ A student with roll number **{reg_roll.strip()}** already exists."
                    )
                else:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Error {exc.response.status_code}: {detail}")
            except Exception as exc:
                st.error(f"Unexpected error during registration: {exc}")

    st.divider()
    st.caption(
        "After registration, go to **Capture & Train** to collect 50–100 face images "
        "and generate face embeddings so the student can be recognised during live attendance."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Capture & Train
# ═════════════════════════════════════════════════════════════════════════════
with tab_capture:
    st.subheader("Capture Face Images & Generate Embeddings")

    # ── Student selector ──────────────────────────────────────────────────────
    with st.spinner("Loading student list…"):
        all_students = st.session_state.students_cache or fetch_students()

    if not all_students:
        st.info("No students registered yet. Use the **Register Student** tab first.")
        st.stop()

    student_options = {
        f"{s.get('name', '')} (Roll: {s.get('roll_number', '')})": s.get("roll_number", "")
        for s in all_students
    }

    selected_label = st.selectbox(
        "Select Student",
        options=list(student_options.keys()),
        key="capture_student_select",
    )
    selected_rn   = student_options.get(selected_label, "")
    selected_name = selected_label.split(" (Roll:")[0] if selected_label else ""

    if not selected_rn:
        st.warning("Please select a valid student.")
        st.stop()

    st.divider()

    # ── Section A: Image Capture ──────────────────────────────────────────────
    st.markdown("### 📸 Step 1 — Capture Face Images")
    st.caption(
        "Activate the webcam and capture up to **100 frames** of the student's face. "
        "Each photo is sent to the server and saved to the dataset folder. "
        "Aim for variety: different angles, lighting, and expressions."
    )

    MAX_FRAMES = 100

    # Per-student capture state keys
    count_key  = f"capture_count_{selected_rn}"
    active_key = f"capture_active_{selected_rn}"

    if count_key  not in st.session_state:
        st.session_state[count_key]  = 0
    if active_key not in st.session_state:
        st.session_state[active_key] = False

    captured_so_far = st.session_state[count_key]

    # ── Progress display ──────────────────────────────────────────────────────
    progress_col, count_col = st.columns([4, 1])
    with progress_col:
        progress_bar = st.progress(
            min(captured_so_far / MAX_FRAMES, 1.0),
            text=f"Captured {captured_so_far} / {MAX_FRAMES} frames",
        )
    with count_col:
        st.metric("Frames", captured_so_far)

    # ── Start / Stop controls ─────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        if st.button(
            "▶ Start Capture",
            key="start_capture_btn",
            disabled=st.session_state[active_key] or captured_so_far >= MAX_FRAMES,
            use_container_width=True,
        ):
            st.session_state[active_key] = True
            st.rerun()

    with btn_col2:
        if st.button(
            "⏹ Stop Capture",
            key="stop_capture_btn",
            disabled=not st.session_state[active_key],
            use_container_width=True,
        ):
            st.session_state[active_key] = False
            st.rerun()

    with btn_col3:
        if st.button("🔄 Reset Count", key="reset_count_btn", use_container_width=False):
            st.session_state[count_key]  = 0
            st.session_state[active_key] = False
            st.rerun()

    # ── Webcam capture loop ───────────────────────────────────────────────────
    if st.session_state[active_key]:
        if captured_so_far >= MAX_FRAMES:
            st.success(f"✅ Reached {MAX_FRAMES} frames. Click **Stop Capture**.")
            st.session_state[active_key] = False
        else:
            st.info(
                f"📷 Capturing for **{selected_name}** (Roll: {selected_rn}). "
                "Click the camera shutter to take each photo."
            )

            # st.camera_input widget — user clicks shutter for each frame
            uploaded_file = st.camera_input(
                f"Take photo {captured_so_far + 1} of {MAX_FRAMES}",
                key=f"camera_{selected_rn}_{captured_so_far}",
            )

            if uploaded_file is not None:
                # Encode the captured image as JPEG base64 and POST to capture endpoint
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    with st.spinner(f"Uploading frame {captured_so_far + 1}…"):
                        resp = api.post(
                            f"/api/students/{selected_rn}/capture",
                            json={"frame": frame_b64},
                        )
                        resp.raise_for_status()

                    result_data = resp.json()
                    saved_count = result_data.get("saved_count", result_data.get("count", captured_so_far + 1))

                    st.session_state[count_key] = saved_count
                    # Update the progress bar immediately
                    progress_bar.progress(
                        min(saved_count / MAX_FRAMES, 1.0),
                        text=f"Captured {saved_count} / {MAX_FRAMES} frames",
                    )

                    if saved_count >= MAX_FRAMES:
                        st.success(
                            f"✅ Captured {MAX_FRAMES} frames for **{selected_name}**. "
                            "Proceed to **Step 2** to generate embeddings."
                        )
                        st.session_state[active_key] = False
                        st.rerun()
                    else:
                        # Rerun to present a fresh camera widget for the next shot
                        st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend.")
                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 422:
                        try:
                            detail = exc.response.json().get("detail", "No face detected in this frame.")
                        except Exception:
                            detail = "No face detected in this frame."
                        st.warning(f"⚠️ Frame rejected: {detail}. Please retake.")
                    else:
                        try:
                            detail = exc.response.json().get("detail", str(exc))
                        except Exception:
                            detail = str(exc)
                        st.error(f"Error {exc.response.status_code}: {detail}")
                except Exception as exc:
                    st.error(f"Unexpected error during capture: {exc}")
    else:
        if captured_so_far == 0:
            st.info("Click **▶ Start Capture** to begin capturing frames.")
        elif captured_so_far >= MAX_FRAMES:
            st.success(f"✅ {MAX_FRAMES} frames already captured. Proceed to Step 2 below.")
        else:
            st.info(
                f"Capture paused at {captured_so_far} frames. "
                "Click **▶ Start Capture** to continue or proceed to Step 2."
            )

    st.divider()

    # ── Section B: Generate Embeddings ───────────────────────────────────────
    st.markdown("### 🧠 Step 2 — Generate Face Embeddings")
    st.caption(
        "Once you have captured enough images (recommended: ≥ 50 frames), "
        "generate embeddings so the student can be recognised during live attendance. "
        "The recogniser will be updated immediately — no server restart is needed."
    )

    if captured_so_far == 0:
        st.warning("Capture at least a few images before generating embeddings.")

    if st.button(
        "⚡ Generate Embeddings",
        key="generate_embeddings_btn",
        disabled=(captured_so_far == 0),
        type="primary",
        use_container_width=False,
    ):
        with st.spinner(f"Generating embeddings for **{selected_name}**… this may take a moment."):
            try:
                resp = api.post(f"/api/students/{selected_rn}/generate-embeddings")
                resp.raise_for_status()
                result = resp.json()

                accepted = result.get("accepted_images", result.get("count", result.get("accepted", 0)))
                warning_msg = result.get("warning", "")

                st.success(
                    f"✅ Embeddings generated successfully for **{selected_name}**. "
                    f"Accepted images: **{accepted}**."
                )
                if warning_msg:
                    st.warning(f"⚠️ {warning_msg}")

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
                st.error(f"Unexpected error during embedding generation: {exc}")
