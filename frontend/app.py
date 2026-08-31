"""
frontend/app.py — Entry point for the AI Attendance Monitoring System Streamlit app.

Responsibilities:
  - Configure global page settings (title, layout, sidebar state).
  - Initialise the shared APIClient singleton in st.session_state.
  - Render sidebar navigation links to all 7 pages.
  - Provide a light/dark theme toggle via custom CSS injection.
  - Show a warning banner when the backend is unreachable.
"""

import streamlit as st
import requests

from utils.api_client import APIClient

# ---------------------------------------------------------------------------
# 1. Page configuration — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 2. Session-state initialisation
# ---------------------------------------------------------------------------
if "api" not in st.session_state:
    st.session_state.api = APIClient(base_url="http://localhost:8000")

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ---------------------------------------------------------------------------
# 3. Theme toggle — inject custom CSS for dark mode
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
    /* ---- Main background & text ---- */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
    /* ---- Cards / containers ---- */
    [data-testid="stMetric"],
    [data-testid="stDataFrame"],
    .stAlert {
        background-color: #1a1f2e !important;
    }
    /* ---- Input widgets ---- */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #262b3a !important;
        color: #fafafa !important;
    }
    /* ---- Buttons ---- */
    .stButton > button {
        background-color: #262b3a;
        color: #fafafa;
        border: 1px solid #444;
    }
    .stButton > button:hover {
        background-color: #363d50;
    }
    /* ---- Dataframe headers ---- */
    .dataframe thead th {
        background-color: #262b3a !important;
        color: #fafafa !important;
    }
    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1f2e;
    }
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
    }
    .stTabs [aria-selected="true"] {
        color: #fafafa !important;
    }
</style>
"""

if st.session_state.theme == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. Sidebar — branding, theme toggle, and navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 AI Attendance System")
    st.divider()

    # --- Theme toggle ---
    current_theme = st.session_state.theme
    toggle_label = "☀️ Switch to Light Mode" if current_theme == "dark" else "🌙 Switch to Dark Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.theme = "light" if current_theme == "dark" else "dark"
        st.rerun()

    st.divider()
    st.markdown("### Navigation")

    # st.page_link() is available in Streamlit ≥ 1.29
    st.page_link("app.py",                              label="🏠 Home",               icon=None)
    st.page_link("pages/1_Dashboard.py",                label="📊 Dashboard")
    st.page_link("pages/2_Live_Attendance.py",          label="📷 Live Attendance")
    st.page_link("pages/3_Student_Management.py",       label="👥 Student Management")
    st.page_link("pages/4_Attendance.py",               label="📋 Attendance Records")
    st.page_link("pages/5_Unknown_Gallery.py",          label="❓ Unknown Gallery")
    st.page_link("pages/6_Reports.py",                  label="📄 Reports")
    st.page_link("pages/7_Analytics.py",                label="📈 Analytics")
    st.page_link("pages/8_Settings.py",                 label="⚙️ Settings")

    st.divider()
    st.caption("v1.0.0 · FastAPI + Streamlit")

# ---------------------------------------------------------------------------
# 5. Backend connectivity check — show warning banner if unreachable
# ---------------------------------------------------------------------------
try:
    health_resp = st.session_state.api.get("/api/analytics/dashboard", timeout=2)
    health_resp.raise_for_status()
    backend_ok = True
except requests.exceptions.ConnectionError:
    backend_ok = False
    st.warning(
        "⚠️ **Backend unreachable.** "
        "Make sure the FastAPI server is running on `http://localhost:8000` "
        "before using any features.",
        icon="⚠️",
    )
except requests.exceptions.Timeout:
    backend_ok = False
    st.warning(
        "⚠️ **Backend timed out.** "
        "The server may be starting up — please refresh in a moment.",
        icon="⏱️",
    )
except Exception:
    # Non-connection errors (e.g. 404, 500) mean the server is at least up
    backend_ok = True

# ---------------------------------------------------------------------------
# 6. Home page content
# ---------------------------------------------------------------------------
st.title("🎓 AI Attendance Monitoring System")
st.markdown(
    "Welcome to the **AI-powered attendance tracking** dashboard. "
    "Use the sidebar to navigate between pages."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📷 **Live Attendance**\nProcess webcam frames for real-time face recognition.")
with col2:
    st.info("👥 **Student Management**\nRegister students and manage their face datasets.")
with col3:
    st.info("📊 **Dashboard**\nView today's attendance summary and key metrics.")

col4, col5, col6 = st.columns(3)
with col4:
    st.info("📋 **Attendance Records**\nBrowse and filter historical attendance data.")
with col5:
    st.info("📈 **Analytics**\nVisualise attendance trends and department statistics.")
with col6:
    st.info("⚙️ **Settings**\nConfigure recognition thresholds and system parameters.")

if backend_ok:
    st.success("✅ Backend connected — `http://localhost:8000`")
