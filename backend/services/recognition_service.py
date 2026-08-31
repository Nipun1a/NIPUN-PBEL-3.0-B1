import sys
import os

# Insert project root so the pre-existing ML modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recognizer import Recognizer
from typing import Optional

_recognizer: Optional[Recognizer] = None


def init_recognizer(settings: dict) -> None:
    """Called once from FastAPI lifespan startup."""
    global _recognizer
    _recognizer = Recognizer(
        recognition_threshold=float(settings.get("recognition_threshold", 0.6)),
        smoothing_window=int(settings.get("stable_frame_count", 4)),
    )


def get_recognizer() -> Recognizer:
    if _recognizer is None:
        raise RuntimeError("Recognizer not initialised. Call init_recognizer() first.")
    return _recognizer


def reload_embeddings() -> int:
    r = get_recognizer()
    r.reload_embeddings()
    return len(r._store)


def process_frame(frame_bytes: bytes):
    """Decode JPEG bytes → numpy array → Recognizer.process_frame → results + annotated."""
    import cv2
    import numpy as np
    if not frame_bytes:
        raise ValueError("Could not decode image bytes to a valid frame.")
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes to a valid frame.")
    return get_recognizer().process_frame(frame)
