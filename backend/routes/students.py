"""
routes/students.py

FastAPI APIRouter for all student-related endpoints.

Prefix (/api/students) is applied externally in main.py.

Endpoints:
  POST   /                         → create student (409 on duplicate roll_number)
  GET    /                         → list students (paginated)
  GET    /search                   → search by ?q= query param
  GET    /{roll_number}            → get single student (404 if missing)
  PUT    /{roll_number}            → update student (404 if missing)
  DELETE /{roll_number}            → delete student + dataset folder + embedding (404 if missing)
  POST   /{roll_number}/capture    → decode base64 frame, validate face, save image; 422 if no image/no face
  POST   /{roll_number}/generate-embeddings → generate embeddings, return accepted_count + warning
  DELETE /{roll_number}/images     → delete all images in student folder

Requirements: 3.1–3.7, 4.1–4.5, 20.2
"""
from __future__ import annotations

import base64

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.database.connection import get_db
from backend.models.recognition import FrameRequest
from backend.models.student import StudentCreate, StudentResponse, StudentUpdate
from backend.services import student_service

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /  → create student
# ---------------------------------------------------------------------------

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_student(
    payload: StudentCreate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Create a new student record.

    Returns 409 if a student with the same roll_number already exists.
    Returns 422 if Pydantic validation fails (handled automatically by FastAPI).

    Requirements: 3.1, 3.2
    """
    try:
        student = await student_service.create_student(payload.model_dump(), db)
        return student
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


# ---------------------------------------------------------------------------
# GET /  → list students (paginated)
# ---------------------------------------------------------------------------

@router.get("/")
async def list_students(
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    page_size: int = Query(default=50, ge=1, le=500, description="Records per page"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a paginated list of all students ordered by name.

    Response shape: { total, page, page_size, records }

    Requirements: 3.3
    """
    return await student_service.list_students(db, page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# GET /search  → search by ?q=
# NOTE: must be declared BEFORE /{roll_number} so FastAPI doesn't treat
#       "search" as a roll_number path parameter.
# ---------------------------------------------------------------------------

@router.get("/search")
async def search_students(
    q: str = Query(default="", description="Substring to search in name or roll_number"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Case-insensitive search on student name or roll_number.

    Returns all matching students (not paginated).

    Requirements: 3.7
    """
    return await student_service.search_students(q, db)


# ---------------------------------------------------------------------------
# GET /{roll_number}  → get single student
# ---------------------------------------------------------------------------

@router.get("/{roll_number}")
async def get_student(
    roll_number: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Fetch a single student by roll_number.

    Returns 404 if the student does not exist.

    Requirements: 3.4
    """
    try:
        return await student_service.get_student(roll_number, db)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# PUT /{roll_number}  → update student
# ---------------------------------------------------------------------------

@router.put("/{roll_number}")
async def update_student(
    roll_number: str,
    payload: StudentUpdate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Update mutable fields on an existing student record.

    Only fields included in the request body are modified.
    Returns 404 if the student does not exist.

    Requirements: 3.5
    """
    try:
        return await student_service.update_student(
            roll_number, payload.model_dump(exclude_none=True), db
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /{roll_number}  → delete student + dataset folder + embedding
# ---------------------------------------------------------------------------

@router.delete("/{roll_number}")
async def delete_student(
    roll_number: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete the student record, their dataset folder, and embedding entry.

    Returns 404 if the student does not exist.

    Requirements: 3.6
    """
    try:
        await student_service.delete_student(roll_number, db)
        return {"message": f"Student {roll_number} deleted successfully"}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# POST /{roll_number}/capture  → decode base64 frame, validate face, save image
# ---------------------------------------------------------------------------

@router.post("/{roll_number}/capture")
async def capture_image(
    roll_number: str,
    payload: FrameRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Accept a base64-encoded JPEG frame, detect a face, and save the image to
    the student's dataset folder (``CollectedImages/{Name}_{RN}/``).

    Expected request body::

        { "frame": "<base64-encoded JPEG>" }

    Returns:
        { "saved_image_count": <int> }  — total images in folder after saving.

    Error responses:
      422 — no frame data provided, invalid base64, cv2 decode failure, or no face detected.
      404 — student not found.

    Requirements: 4.1, 4.4, 20.2
    """
    frame_data: str = payload.frame
    if not frame_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No frame data provided.",
        )

    try:
        frame_bytes = base64.b64decode(frame_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 frame data.",
        )

    # Verify student exists before attempting to decode / save
    try:
        student = await student_service.get_student(roll_number, db)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    try:
        count = await student_service.capture_image(
            roll_number, student["name"], frame_bytes, db
        )
        return {"saved_image_count": count}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )


# ---------------------------------------------------------------------------
# POST /{roll_number}/generate-embeddings  → generate embeddings
# ---------------------------------------------------------------------------

@router.post("/{roll_number}/generate-embeddings")
async def generate_embeddings(
    roll_number: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Generate face embeddings from the student's saved dataset images.

    Hot-reloads the recognition singleton so new embeddings take effect
    immediately without a server restart.

    Returns::

        { "accepted_count": <int>, "warning": "<str>" }

    ``warning`` is a non-empty string when fewer than 10 images were accepted
    by quality filtering, or when the dataset folder was empty.

    Error responses:
      404 — student not found, or dataset folder does not exist.

    Requirements: 4.2, 4.3
    """
    try:
        result = await student_service.generate_embeddings(roll_number, db)
        return result
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /{roll_number}/images  → delete all images in student folder
# ---------------------------------------------------------------------------

@router.delete("/{roll_number}/images")
async def delete_images(
    roll_number: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete all ``.jpg`` images in the student's dataset folder.

    The folder itself is preserved. Returns the number of files deleted.

    Returns::

        { "message": "Deleted N image(s)", "deleted_count": <int> }

    Error responses:
      404 — student not found.

    Requirements: 4.5
    """
    try:
        deleted = await student_service.delete_images(roll_number, db)
        return {"message": f"Deleted {deleted} image(s)", "deleted_count": deleted}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
