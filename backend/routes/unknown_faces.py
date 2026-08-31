"""
routes/unknown_faces.py

FastAPI APIRouter for unknown face management endpoints.

Prefix (/api/unknown-faces) is applied externally in main.py.

Endpoints:
  GET    /               → paginated list with filters (Req 21.1)
  GET    /stats          → aggregated statistics (Req 21.7)  ← BEFORE /{id}
  DELETE /bulk           → bulk delete by list of ids (Req 21.4)  ← BEFORE /{id}
  POST   /{id}/register  → register unknown face as new student (Req 21.5, 21.6)
  GET    /{id}           → single record by id (Req 21.2)
  DELETE /{id}           → delete record + JPEG file (Req 21.3)

CRITICAL route ordering:
  /stats and /bulk MUST be declared BEFORE /{id} so FastAPI does not
  treat the literal strings "stats" or "bulk" as integer id path parameters.

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7
"""
from __future__ import annotations

from typing import List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.models.student import StudentCreate
from backend.services import unknown_faces_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request model for bulk delete
# ---------------------------------------------------------------------------

class BulkDeleteRequest(BaseModel):
    ids: List[int]


# ---------------------------------------------------------------------------
# 1. GET /  → paginated list with optional filters
# ---------------------------------------------------------------------------

@router.get("/", status_code=status.HTTP_200_OK)
async def list_unknown_faces(
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Records per page (max 100)"),
    date: Optional[str] = Query(
        default=None,
        description="Filter by exact date of the timestamp column (YYYY-MM-DD)",
    ),
    start_date: Optional[str] = Query(
        default=None,
        description="Inclusive start date for range filter (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="Inclusive end date for range filter (YYYY-MM-DD)",
    ),
    min_confidence: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence_score filter (0.0–1.0)",
    ),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return a paginated list of unknown face records.

    Optional query parameters allow filtering by date, date range, and
    minimum confidence score.

    Response shape::

        {
            "total": <int>,
            "page": <int>,
            "page_size": <int>,
            "records": [
                {
                    "id": <int>,
                    "timestamp": "<ISO-8601 string>",
                    "confidence_score": <float>,
                    "image_data": "<base64-JPEG or empty string>",
                    "image_path": "<string>",
                    "created_at": "<ISO-8601 string>"
                },
                ...
            ]
        }

    Requirements: 21.1
    """
    return await unknown_faces_service.get_paginated(
        db=db,
        date=date,
        start_date=start_date,
        end_date=end_date,
        min_confidence=min_confidence,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# 2. GET /stats  → aggregated statistics
#    MUST be declared BEFORE GET /{id} to avoid routing conflict.
# ---------------------------------------------------------------------------

@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_unknown_faces_stats(
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return aggregated statistics for the unknown_faces table.

    Response shape::

        {
            "total_logged": <int>,
            "logged_today": <int>,
            "logged_this_week": <int>,
            "average_confidence_score": <float>
        }

    Requirements: 21.7
    """
    return await unknown_faces_service.get_stats(db=db)


# ---------------------------------------------------------------------------
# 3. DELETE /bulk  → bulk delete by list of ids
#    MUST be declared BEFORE DELETE /{id} to avoid routing conflict.
# ---------------------------------------------------------------------------

@router.delete("/bulk", status_code=status.HTTP_200_OK)
async def bulk_delete_unknown_faces(
    body: BulkDeleteRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete multiple unknown face records (and their JPEG crop files) by id.

    Request body::

        { "ids": [1, 2, 3] }

    Response::

        { "deleted_count": <int> }

    Requirements: 21.4
    """
    deleted_count = await unknown_faces_service.bulk_delete(ids=body.ids, db=db)
    return {"deleted_count": deleted_count}


# ---------------------------------------------------------------------------
# 4. POST /{id}/register  → register unknown face as a new student
# ---------------------------------------------------------------------------

@router.post("/{record_id}/register", status_code=status.HTTP_201_CREATED)
async def register_unknown_face(
    record_id: int,
    payload: StudentCreate,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Register a new student using the unknown face crop identified by *record_id*.

    Steps performed by the service:
    1. Fetches the unknown face record (HTTP 404 if not found).
    2. Verifies the roll_number does not already exist (HTTP 409 if duplicate).
    3. Creates a new student row in the database.
    4. Copies the JPEG crop to ``CollectedImages/{Name}_{RollNumber}/``.
    5. Generates embeddings from the saved image.
    6. Persists the embedding and hot-reloads the Recognizer.

    Response (HTTP 201)::

        {
            "student": { <student fields> },
            "warning": "<optional warning string>"
        }

    Requirements: 21.5, 21.6
    """
    student_data = payload.model_dump()

    try:
        result = await unknown_faces_service.register_from_unknown(
            id=record_id,
            student_data=student_data,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        error_msg = str(exc)
        if "409" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg,
        ) from exc

    return result


# ---------------------------------------------------------------------------
# 5. GET /{id}  → single record by id
# ---------------------------------------------------------------------------

@router.get("/{record_id}", status_code=status.HTTP_200_OK)
async def get_unknown_face(
    record_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Return the unknown face record identified by *record_id*.

    Response shape::

        {
            "id": <int>,
            "timestamp": "<ISO-8601 string>",
            "confidence_score": <float>,
            "image_data": "<base64-JPEG or empty string>",
            "image_path": "<string>",
            "created_at": "<ISO-8601 string>"
        }

    Returns HTTP 404 if no record with that id exists.

    Requirements: 21.2
    """
    try:
        return await unknown_faces_service.get_by_id(id=record_id, db=db)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# 6. DELETE /{id}  → delete record + JPEG crop file
# ---------------------------------------------------------------------------

@router.delete("/{record_id}", status_code=status.HTTP_200_OK)
async def delete_unknown_face(
    record_id: int,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Delete the unknown face record identified by *record_id* and its
    associated JPEG crop file from disk.

    Returns HTTP 200 with a confirmation message on success.
    Returns HTTP 404 if no record with that id exists.

    Requirements: 21.3
    """
    try:
        await unknown_faces_service.delete_one(id=record_id, db=db)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return {"message": f"Unknown face record {record_id} deleted successfully."}
