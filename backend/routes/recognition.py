"""
recognition.py — Recognition API routes.

Endpoints:
    POST /process-frame       — Decode base64 frame, run ML pipeline, mark attendance.
    POST /reload-embeddings   — Hot-reload face embeddings from disk.
    GET  /stream              — MJPEG stream from configured camera.

Requirements: 5.1–5.6, 19.1, 20.3, 20.4
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone

import cv2
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.database.connection import get_db
from backend.models.recognition import FrameRequest, FrameResult, RecognitionResponse
from backend.services import recognition_service, attendance_service, settings_service

router = APIRouter()

# Maximum allowed payload size (10 MB decoded bytes). Requirement 5.3.
MAX_FRAME_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# POST /process-frame
# ---------------------------------------------------------------------------

@router.post("/process-frame", response_model=RecognitionResponse)
async def process_frame(
    payload: FrameRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Accept a base64-encoded JPEG frame, run the full ML recognition pipeline,
    optionally mark attendance for recognised students, and return annotated
    results.

    Raises:
        413 — decoded payload exceeds 10 MB.
        422 — invalid base64 or non-decodable image.
    """
    # 1. Decode base64 → bytes. Requirement 5.2.
    try:
        frame_bytes = base64.b64decode(payload.frame)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid base64 encoding.")

    # 2. Size guard — checked after decode so we measure real bytes. Req 5.3.
    if len(frame_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Payload exceeds 10 MB limit.",
        )

    # 3. Decode image bytes through ML pipeline. Req 5.1.
    try:
        results, annotated_frame = recognition_service.process_frame(frame_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 4. Read cooldown setting once per request. Req 5.5.
    settings = await settings_service.load_settings_from_db(db)
    cooldown = int(settings.get("cooldown_period_seconds", 300))

    # 5. Per-result attendance logic. Req 5.4, 5.5, 5.6.
    frame_results: list[FrameResult] = []

    for r in results:
        attendance_marked = False
        duplicate = False

        if r.recognition_status == "Known":
            # Duplicate cooldown check before writing. Req 5.5.
            is_dup = await attendance_service.is_duplicate(r.roll_number, cooldown, db)
            if is_dup:
                duplicate = True
            else:
                await attendance_service.mark_attendance(
                    r.roll_number, r.name, r.confidence_score, db
                )
                attendance_marked = True

        elif r.recognition_status == "Unknown":
            # Log unknown face to the unknown_faces table. Req 5.6, 19.1.
            now_iso = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT INTO unknown_faces "
                "(timestamp, confidence_score, image_path, created_at) "
                "VALUES (?, ?, ?, ?)",
                (now_iso, r.confidence_score, "", now_iso),
            )
            await db.commit()

        frame_results.append(
            FrameResult(
                name=r.name,
                roll_number=r.roll_number,
                confidence_score=r.confidence_score,
                recognition_status=r.recognition_status,
                bounding_box=tuple(r.bounding_box),  # type: ignore[arg-type]
                attendance_marked=attendance_marked,
                duplicate=duplicate,
            )
        )

    # 6. Encode annotated frame as base64 JPEG for the response. Req 20.3, 20.4.
    if annotated_frame is not None:
        _, buf = cv2.imencode(".jpg", annotated_frame)
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    else:
        annotated_b64 = ""

    return RecognitionResponse(results=frame_results, annotated_frame=annotated_b64)


# ---------------------------------------------------------------------------
# POST /reload-embeddings
# ---------------------------------------------------------------------------

@router.post("/reload-embeddings")
async def reload_embeddings():
    """
    Hot-reload face embeddings from disk into the live Recognizer singleton.
    Returns the number of students now loaded.
    """
    try:
        count = recognition_service.reload_embeddings()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"message": "Embeddings reloaded successfully.", "student_count": count}


# ---------------------------------------------------------------------------
# GET /stream
# ---------------------------------------------------------------------------

@router.get("/stream")
async def stream():
    """
    Return a multipart/x-mixed-replace MJPEG stream captured from the
    camera_index configured in BackendSettings.

    The camera index defaults to 0 when not explicitly set.
    """
    from backend.config import settings as app_settings

    camera_index: int = getattr(app_settings, "camera_index", 0)

    def generate():
        cap = cv2.VideoCapture(camera_index)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, buf = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
        finally:
            cap.release()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
