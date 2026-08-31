"""
Unknown Face Gallery page — AI Attendance Monitoring System

Displays a grid of captured unknown-face thumbnails.  Supports:
  - Stats header (total logged, logged today, average confidence)
  - Date-range filter, min-confidence slider, "Today only" toggle
  - 4-column thumbnail grid with timestamp and confidence labels
  - Per-face expander: full-size crop, detail info, Register / Delete actions
  - Register-as-new-student inline form (POST /api/unknown-faces/{id}/register)
  - Single delete with confirmation (DELETE /api/unknown-faces/{id})
  - Select-all checkbox + bulk delete (DELETE /api/unknown-faces/bulk)
  - Empty-state message when no faces match the current filters

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9
"""

import base64
from datetime import date, datetime

import requests
import streamlit as st

from utils.api_client import APIClient

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unknown Gallery — Attendance Monitor",
    page_icon="❓",
    layout="wide",
)

# ── Ensure API client is available ────────────────────────────────────────────
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

api: APIClient = st.session_state.api

# ── Session-state initialisation ──────────────────────────────────────────────
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids: set = set()

if "gallery_records" not in st.session_state:
    st.session_state.gallery_records = []

if "gallery_needs_refresh" not in st.session_state:
    st.session_state.gallery_needs_refresh = True

# ── Page title ────────────────────────────────────────────────────────────────
st.title("❓ Unknown Face Gallery")

# ── Stats header ──────────────────────────────────────────────────────────────
try:
    stats_resp = api.get("/api/unknown-faces/stats")
    stats_resp.raise_for_status()
    stats = stats_resp.json()
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
    stats = {}
except Exception as exc:
    st.error(f"Unexpected error loading stats: {exc}")
    stats = {}

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    st.metric("Total Logged", stats.get("total_logged", "—"))
with stat_col2:
    st.metric("Logged Today", stats.get("logged_today", "—"))
with stat_col3:
    st.metric("Logged This Week", stats.get("logged_this_week", "—"))
with stat_col4:
    avg_conf = stats.get("average_confidence_score")
    avg_display = f"{avg_conf:.2f}" if avg_conf is not None else "—"
    st.metric("Avg Confidence", avg_display)

st.divider()

# ── Filter controls ───────────────────────────────────────────────────────────
st.subheader("🔍 Filters")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])

today = date.today()

with filter_col1:
    start_date = st.date_input("Start Date", value=None, key="uf_start_date")

with filter_col2:
    end_date = st.date_input("End Date", value=None, key="uf_end_date")

with filter_col3:
    min_confidence = st.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="uf_min_confidence",
    )

with filter_col4:
    st.write("")  # vertical spacing
    today_only = st.checkbox("Today Only", key="uf_today_only")

# Apply "Today only" toggle — override date range inputs
if today_only:
    effective_start = today
    effective_end = today
else:
    effective_start = start_date
    effective_end = end_date

# Build query params
gallery_params: dict = {}
if effective_start:
    gallery_params["start_date"] = effective_start.isoformat()
if effective_end:
    gallery_params["end_date"] = effective_end.isoformat()
if min_confidence > 0.0:
    gallery_params["min_confidence"] = min_confidence

# ── Fetch gallery records ─────────────────────────────────────────────────────
gallery_records = []

with st.spinner("Loading gallery..."):
    try:
        gallery_resp = api.get("/api/unknown-faces", params=gallery_params)
        gallery_resp.raise_for_status()
        gallery_data = gallery_resp.json()
        # API may return a list directly or a paginated dict
        if isinstance(gallery_data, list):
            gallery_records = gallery_data
        else:
            gallery_records = gallery_data.get("records", [])
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
    except Exception as exc:
        st.error(f"Unexpected error loading gallery: {exc}")

st.divider()

# ── Bulk-action bar ───────────────────────────────────────────────────────────
bulk_col1, bulk_col2, bulk_col3 = st.columns([1, 2, 4])

all_ids = {rec["id"] for rec in gallery_records}

with bulk_col1:
    select_all = st.checkbox(
        "Select All",
        value=(len(st.session_state.selected_ids) == len(all_ids) and len(all_ids) > 0),
        key="uf_select_all",
        disabled=(len(all_ids) == 0),
    )
    if select_all:
        st.session_state.selected_ids = set(all_ids)
    elif not select_all and len(st.session_state.selected_ids) == len(all_ids) and len(all_ids) > 0:
        # Only clear if the user explicitly unchecked; avoid clearing on page load
        st.session_state.selected_ids = set()

with bulk_col2:
    selected_count = len(st.session_state.selected_ids)
    delete_selected_label = (
        f"🗑️ Delete Selected ({selected_count})"
        if selected_count > 0
        else "🗑️ Delete Selected"
    )
    delete_selected_clicked = st.button(
        delete_selected_label,
        disabled=(selected_count == 0),
        type="primary",
        key="uf_delete_selected_btn",
    )

if delete_selected_clicked and selected_count > 0:
    st.warning(
        f"⚠️ About to delete **{selected_count}** selected face(s). "
        "This cannot be undone."
    )
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("✅ Confirm Bulk Delete", key="uf_confirm_bulk"):
            with st.spinner("Deleting selected faces..."):
                try:
                    bulk_resp = api.delete(
                        "/api/unknown-faces/bulk",
                        json={"ids": list(st.session_state.selected_ids)},
                    )
                    bulk_resp.raise_for_status()
                    result = bulk_resp.json()
                    deleted_count = result.get("deleted_count", selected_count)
                    st.success(f"✅ Deleted {deleted_count} face record(s).")
                    st.session_state.selected_ids = set()
                    st.rerun()
                except requests.exceptions.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Error {exc.response.status_code}: {detail}")
                except Exception as exc:
                    st.error(f"Unexpected error during bulk delete: {exc}")
    with confirm_col2:
        if st.button("❌ Cancel", key="uf_cancel_bulk"):
            st.rerun()

st.divider()

# ── Gallery grid ──────────────────────────────────────────────────────────────
if not gallery_records:
    st.info("No unknown faces found matching the current filters.")
else:
    COLS = 4
    rows = [gallery_records[i: i + COLS] for i in range(0, len(gallery_records), COLS)]

    for row in rows:
        cols = st.columns(COLS)
        for col, record in zip(cols, row):
            face_id = record["id"]
            timestamp_raw = record.get("timestamp", "")
            confidence = record.get("confidence_score", 0.0)
            image_data_b64 = record.get("image_data", "")

            # Format timestamp for display
            try:
                ts_dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                ts_display = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_display = timestamp_raw

            with col:
                # ── Thumbnail image ───────────────────────────────────────
                if image_data_b64:
                    try:
                        img_bytes = base64.b64decode(image_data_b64)
                        st.image(img_bytes, use_container_width=True)
                    except Exception:
                        st.image(
                            "https://via.placeholder.com/150?text=No+Image",
                            use_container_width=True,
                        )
                else:
                    st.caption("🚫 No image")

                # ── Caption row ───────────────────────────────────────────
                st.caption(f"🕐 {ts_display}")
                st.caption(f"📊 Confidence: {confidence:.2f}")

                # ── Per-face checkbox for bulk selection ──────────────────
                is_checked = st.checkbox(
                    f"Select #{face_id}",
                    value=(face_id in st.session_state.selected_ids),
                    key=f"uf_select_{face_id}",
                )
                if is_checked:
                    st.session_state.selected_ids.add(face_id)
                else:
                    st.session_state.selected_ids.discard(face_id)

                # ── Detail expander ───────────────────────────────────────
                with st.expander(f"Details — #{face_id}", expanded=False):
                    # Full-size crop
                    if image_data_b64:
                        try:
                            img_bytes = base64.b64decode(image_data_b64)
                            st.image(img_bytes, use_container_width=True)
                        except Exception:
                            st.warning("Could not decode image data.")
                    else:
                        st.info("No image available for this record.")

                    st.markdown(f"**Timestamp:** {ts_display}")
                    st.markdown(f"**Confidence Score:** {confidence:.4f}")
                    st.markdown(f"**Record ID:** {face_id}")

                    st.divider()

                    # ── Delete button ─────────────────────────────────────
                    if st.button(
                        "🗑️ Delete",
                        key=f"uf_delete_btn_{face_id}",
                        type="secondary",
                    ):
                        st.session_state[f"uf_confirm_delete_{face_id}"] = True

                    if st.session_state.get(f"uf_confirm_delete_{face_id}", False):
                        st.warning("⚠️ This will permanently delete this face record.")
                        del_conf_col1, del_conf_col2 = st.columns(2)
                        with del_conf_col1:
                            if st.button(
                                "✅ Confirm Delete",
                                key=f"uf_confirm_delete_yes_{face_id}",
                            ):
                                try:
                                    del_resp = api.delete(
                                        f"/api/unknown-faces/{face_id}"
                                    )
                                    del_resp.raise_for_status()
                                    st.success("✅ Face record deleted.")
                                    st.session_state.selected_ids.discard(face_id)
                                    st.session_state[
                                        f"uf_confirm_delete_{face_id}"
                                    ] = False
                                    st.rerun()
                                except requests.exceptions.HTTPError as exc:
                                    try:
                                        detail = exc.response.json().get(
                                            "detail", str(exc)
                                        )
                                    except Exception:
                                        detail = str(exc)
                                    st.error(
                                        f"Error {exc.response.status_code}: {detail}"
                                    )
                                except Exception as exc:
                                    st.error(f"Unexpected error: {exc}")
                        with del_conf_col2:
                            if st.button(
                                "❌ Cancel",
                                key=f"uf_confirm_delete_no_{face_id}",
                            ):
                                st.session_state[
                                    f"uf_confirm_delete_{face_id}"
                                ] = False
                                st.rerun()

                    st.divider()

                    # ── Register as New Student ───────────────────────────
                    st.markdown("#### 👤 Register as New Student")
                    with st.form(key=f"uf_register_form_{face_id}"):
                        reg_roll = st.text_input(
                            "Roll Number *",
                            key=f"uf_roll_{face_id}",
                            placeholder="e.g. 101",
                        )
                        reg_name = st.text_input(
                            "Full Name *",
                            key=f"uf_name_{face_id}",
                            placeholder="e.g. Rahul Sharma",
                        )
                        reg_dept = st.text_input(
                            "Department",
                            key=f"uf_dept_{face_id}",
                            placeholder="e.g. CS",
                        )
                        reg_email = st.text_input(
                            "Email",
                            key=f"uf_email_{face_id}",
                            placeholder="e.g. rahul@example.com",
                        )
                        reg_phone = st.text_input(
                            "Phone",
                            key=f"uf_phone_{face_id}",
                            placeholder="e.g. 9876543210",
                        )
                        register_submitted = st.form_submit_button(
                            "📝 Register Student", type="primary"
                        )

                    if register_submitted:
                        if not reg_roll.strip() or not reg_name.strip():
                            st.error("Roll Number and Full Name are required.")
                        else:
                            payload = {
                                "roll_number": reg_roll.strip(),
                                "name": reg_name.strip(),
                                "department": reg_dept.strip(),
                                "email": reg_email.strip(),
                                "phone": reg_phone.strip(),
                            }
                            with st.spinner("Registering student..."):
                                try:
                                    reg_resp = api.post(
                                        f"/api/unknown-faces/{face_id}/register",
                                        json=payload,
                                    )
                                    if reg_resp.status_code == 201:
                                        reg_data = reg_resp.json()
                                        student_obj = reg_data.get("student", {})
                                        student_name = student_obj.get(
                                            "name", reg_name.strip()
                                        )
                                        warning_msg = reg_data.get("warning", "")
                                        st.success(
                                            f"✅ Student **{student_name}** registered successfully!"
                                        )
                                        if warning_msg:
                                            st.warning(f"⚠️ {warning_msg}")
                                        # Refresh gallery
                                        st.session_state.selected_ids.discard(face_id)
                                        st.rerun()
                                    elif reg_resp.status_code == 409:
                                        try:
                                            detail = reg_resp.json().get(
                                                "detail",
                                                "A student with this roll number already exists.",
                                            )
                                        except Exception:
                                            detail = "A student with this roll number already exists."
                                        st.error(f"❌ Conflict (409): {detail}")
                                    elif reg_resp.status_code == 404:
                                        try:
                                            detail = reg_resp.json().get(
                                                "detail",
                                                "Unknown face record not found.",
                                            )
                                        except Exception:
                                            detail = "Unknown face record not found."
                                        st.error(f"❌ Not Found (404): {detail}")
                                    else:
                                        reg_resp.raise_for_status()
                                except requests.exceptions.HTTPError as exc:
                                    try:
                                        detail = exc.response.json().get(
                                            "detail", str(exc)
                                        )
                                    except Exception:
                                        detail = str(exc)
                                    st.error(
                                        f"Error {exc.response.status_code}: {detail}"
                                    )
                                except Exception as exc:
                                    st.error(f"Unexpected error: {exc}")
