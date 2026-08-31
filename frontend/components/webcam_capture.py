import base64
import io
from typing import Optional

import streamlit as st
from PIL import Image


def capture_and_process(api, endpoint: str) -> Optional[dict]:
    """Capture a frame from the webcam and POST it to the given API endpoint.

    Workflow:
        st.camera_input → PIL open → JPEG encode (quality=85) → base64 → POST → return JSON

    Args:
        api:      An APIClient instance (has a .post(path, json=...) method).
        endpoint: The API path to POST the encoded frame to,
                  e.g. "/api/recognition/process-frame".

    Returns:
        Parsed JSON response dict on success, or None if no frame was captured.
    """
    uploaded_file = st.camera_input("Capture frame")
    if uploaded_file is None:
        return None

    img = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = api.post(endpoint, json={"frame": frame_b64})
    response.raise_for_status()
    return response.json()
