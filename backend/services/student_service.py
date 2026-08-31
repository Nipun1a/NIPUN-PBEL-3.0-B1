"""
student_service.py

Async service layer for student CRUD operations, image capture, embedding
generation, and dataset folder management.

All SQL queries use parameterised placeholders (``?``) — no raw string
interpolation — to prevent SQL injection.

ML modules (EmbeddingGenerator, EmbeddingStore) are imported from the project
root via sys.path.insert so the pre-existing ML pipeline is never modified.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.5
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root (one level above backend/) is on sys.path so that
# the pre-existing ML modules are importable without modification.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from embedding_generator import EmbeddingGenerator, EmbeddingStore, StudentRecord

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level EmbeddingGenerator singleton — loaded once at import time.
# This avoids re-loading the heavy ArcFace model on every request.
# ---------------------------------------------------------------------------
_embedding_generator: EmbeddingGenerator | None = None


def _get_embedding_generator() -> EmbeddingGenerator:
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


# ---------------------------------------------------------------------------
# Helper — folder path for a student's dataset images
# ---------------------------------------------------------------------------

def _student_folder(name: str, roll_number: str) -> str:
    """Return the absolute path to ``CollectedImages/{Name}_{RN}/``."""
    folder_name = f"{name}_{roll_number}"
    return os.path.join(settings.dataset_root, folder_name)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. create_student
# ---------------------------------------------------------------------------

async def create_student(student_data: dict, db) -> dict:
    """
    Insert a new student row into the ``students`` table.

    Parameters
    ----------
    student_data : dict
        Must contain ``roll_number`` and ``name``; may contain
        ``department``, ``email``, ``phone``.
    db : aiosqlite.Connection

    Returns
    -------
    dict
        The newly created student record (all columns).

    Raises
    ------
    ValueError
        If ``roll_number`` already exists in the database.

    Requirements: 3.1, 3.2
    """
    roll_number = student_data["roll_number"]
    name = student_data["name"]
    department = student_data.get("department", "")
    email = student_data.get("email", "")
    phone = student_data.get("phone", "")
    now = _now_iso()

    # Check for duplicate roll_number
    async with db.execute(
        "SELECT roll_number FROM students WHERE roll_number = ?",
        (roll_number,),
    ) as cursor:
        existing = await cursor.fetchone()

    if existing is not None:
        raise ValueError(f"Student with roll_number '{roll_number}' already exists.")

    await db.execute(
        """
        INSERT INTO students (roll_number, name, department, email, phone, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (roll_number, name, department, email, phone, now, now),
    )
    await db.commit()

    return {
        "roll_number": roll_number,
        "name": name,
        "department": department,
        "email": email,
        "phone": phone,
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# 2. get_student
# ---------------------------------------------------------------------------

async def get_student(roll_number: str, db) -> dict:
    """
    Fetch a single student by roll number.

    Parameters
    ----------
    roll_number : str
    db : aiosqlite.Connection

    Returns
    -------
    dict
        The student record.

    Raises
    ------
    KeyError
        If no student with ``roll_number`` exists.

    Requirements: 3.4
    """
    async with db.execute(
        "SELECT roll_number, name, department, email, phone, created_at, updated_at "
        "FROM students WHERE roll_number = ?",
        (roll_number,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise KeyError(f"Student '{roll_number}' not found.")

    return dict(row)


# ---------------------------------------------------------------------------
# 3. list_students
# ---------------------------------------------------------------------------

async def list_students(db, page: int = 1, page_size: int = 50) -> dict:
    """
    Return a paginated list of all students ordered by name.

    Parameters
    ----------
    db : aiosqlite.Connection
    page : int
        1-indexed page number (default 1).
    page_size : int
        Number of records per page (default 50).

    Returns
    -------
    dict
        ``{ total, page, page_size, records }``

    Requirements: 3.3
    """
    page = max(1, page)
    page_size = max(1, page_size)
    offset = (page - 1) * page_size

    async with db.execute("SELECT COUNT(*) FROM students") as cursor:
        total_row = await cursor.fetchone()
    total = total_row[0] if total_row else 0

    async with db.execute(
        "SELECT roll_number, name, department, email, phone, created_at, updated_at "
        "FROM students ORDER BY name ASC LIMIT ? OFFSET ?",
        (page_size, offset),
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": [dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# 4. search_students
# ---------------------------------------------------------------------------

async def search_students(query: str, db) -> list[dict]:
    """
    Case-insensitive search on ``name`` OR ``roll_number``.

    Parameters
    ----------
    query : str
        Substring to search for.
    db : aiosqlite.Connection

    Returns
    -------
    list[dict]
        Matching student records.

    Requirements: 3.7
    """
    pattern = f"%{query}%"
    async with db.execute(
        "SELECT roll_number, name, department, email, phone, created_at, updated_at "
        "FROM students "
        "WHERE LOWER(name) LIKE LOWER(?) OR LOWER(roll_number) LIKE LOWER(?) "
        "ORDER BY name ASC",
        (pattern, pattern),
    ) as cursor:
        rows = await cursor.fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 5. update_student
# ---------------------------------------------------------------------------

async def update_student(roll_number: str, updates: dict, db) -> dict:
    """
    Update mutable fields on an existing student record.

    Only fields present in ``updates`` are modified; ``updated_at`` is always
    refreshed.

    Parameters
    ----------
    roll_number : str
    updates : dict
        May contain any subset of ``name``, ``department``, ``email``, ``phone``.
    db : aiosqlite.Connection

    Returns
    -------
    dict
        The full updated student record.

    Raises
    ------
    KeyError
        If no student with ``roll_number`` exists.

    Requirements: 3.5
    """
    # Verify the student exists
    await get_student(roll_number, db)

    allowed_fields = {"name", "department", "email", "phone"}
    set_clauses: list[str] = []
    values: list[Any] = []

    for field, value in updates.items():
        if field in allowed_fields and value is not None:
            set_clauses.append(f"{field} = ?")
            values.append(value)

    now = _now_iso()
    set_clauses.append("updated_at = ?")
    values.append(now)
    values.append(roll_number)

    if len(set_clauses) > 1:  # at least one real field + updated_at
        sql = f"UPDATE students SET {', '.join(set_clauses)} WHERE roll_number = ?"
        await db.execute(sql, values)
        await db.commit()

    return await get_student(roll_number, db)


# ---------------------------------------------------------------------------
# 6. delete_student
# ---------------------------------------------------------------------------

async def delete_student(roll_number: str, db) -> None:
    """
    Delete a student and all associated data.

    Steps:
      1. Fetch the student record (raises KeyError if not found).
      2. DELETE the database row.
      3. Remove ``CollectedImages/{Name}_{RN}/`` via ``shutil.rmtree`` (if it
         exists).
      4. Remove the student from the EmbeddingStore pickle file.

    Parameters
    ----------
    roll_number : str
    db : aiosqlite.Connection

    Raises
    ------
    KeyError
        If no student with ``roll_number`` exists.

    Requirements: 3.6
    """
    student = await get_student(roll_number, db)
    name = student["name"]

    # 1. Delete the database row
    await db.execute(
        "DELETE FROM students WHERE roll_number = ?",
        (roll_number,),
    )
    await db.commit()

    # 2. Remove the dataset folder
    folder = _student_folder(name, roll_number)
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
            logger.info("Removed dataset folder: %s", folder)
        except OSError as exc:
            logger.warning("Could not remove dataset folder %s: %s", folder, exc)

    # 3. Remove from EmbeddingStore (no-op if not present)
    try:
        EmbeddingStore.delete_student(roll_number, settings.embeddings_file)
        logger.info("Removed embedding entry for roll_number '%s'", roll_number)
    except Exception as exc:
        logger.warning("Could not remove embedding for '%s': %s", roll_number, exc)


# ---------------------------------------------------------------------------
# 7. capture_image
# ---------------------------------------------------------------------------

async def capture_image(
    roll_number: str,
    name: str,
    frame_bytes: bytes,
    db,  # noqa: ARG001 — kept for API consistency; may be used for audit later
) -> int:
    """
    Decode a JPEG frame, detect a face, and save it to the student folder.

    The saved filename follows the pattern ``{Name}_{seq:03d}.jpg`` where
    ``seq`` is determined by the count of existing ``.jpg`` files in the
    folder (1-indexed so that the first image is ``001``).

    Parameters
    ----------
    roll_number : str
    name : str
    frame_bytes : bytes
        Raw JPEG bytes.
    db : aiosqlite.Connection

    Returns
    -------
    int
        Total number of ``.jpg`` images now in the student folder after saving.

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as an image, or if no face is detected.

    Requirements: 4.1
    """
    # Decode JPEG bytes → numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame bytes into a valid image.")

    # Face detection — import FaceDetector lazily (ML module)
    from face_detector import FaceDetector  # noqa: PLC0415

    detector = FaceDetector()
    detections = detector.detect(frame)
    if not detections:
        raise ValueError("No face detected in the captured frame.")

    # Ensure the student folder exists
    folder = _student_folder(name, roll_number)
    os.makedirs(folder, exist_ok=True)

    # Determine the next sequence number
    existing = [
        f for f in os.listdir(folder) if f.lower().endswith(".jpg")
    ]
    seq = len(existing) + 1

    filename = f"{name}_{seq:03d}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    logger.info("Saved captured image: %s", filepath)

    return seq  # equals total saved count after this write


# ---------------------------------------------------------------------------
# 8. generate_embeddings
# ---------------------------------------------------------------------------

async def generate_embeddings(roll_number: str, db) -> dict:
    """
    Generate face embeddings for a student's dataset folder and persist them.

    Steps:
      1. Fetch the student record.
      2. Run ``EmbeddingGenerator.generate_embeddings`` on the dataset folder.
      3. Aggregate individual embeddings into a representative embedding using
         ``EmbeddingGenerator._aggregate_embeddings``.
      4. Upsert the student's entry in ``EmbeddingStore``.
      5. Reload the in-memory recognizer singleton.
      6. Return ``{ accepted_count, warning }``; ``warning`` is non-empty when
         fewer than 10 images were accepted.

    Parameters
    ----------
    roll_number : str
    db : aiosqlite.Connection

    Returns
    -------
    dict
        ``{ "accepted_count": int, "warning": str }``

    Raises
    ------
    KeyError
        If the student does not exist.
    FileNotFoundError
        If the student's dataset folder does not exist or is empty.

    Requirements: 4.2, 4.3
    """
    student = await get_student(roll_number, db)
    name = student["name"]
    folder = _student_folder(name, roll_number)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder for student '{roll_number}' not found: {folder}"
        )

    generator = _get_embedding_generator()

    # Generate per-image embeddings (quality-gated)
    individual_embeddings = generator.generate_embeddings(folder)
    accepted_count = len(individual_embeddings)

    warning = ""
    representative: np.ndarray | None = None

    if accepted_count == 0:
        warning = (
            "No images passed quality filtering. "
            "Embedding store was not updated. "
            "Please capture more images."
        )
        logger.warning(
            "generate_embeddings: no accepted images for roll_number '%s'", roll_number
        )
        return {"accepted_count": 0, "warning": warning}

    # Aggregate into a single representative embedding
    representative = generator._aggregate_embeddings(
        individual_embeddings, name, roll_number
    )
    if representative is None:
        warning = (
            "Aggregated embedding produced a zero-norm vector. "
            "Embedding store was not updated."
        )
        logger.error(
            "generate_embeddings: zero-norm representative for '%s'", roll_number
        )
        return {"accepted_count": accepted_count, "warning": warning}

    # Build StudentRecord and upsert into EmbeddingStore
    record = StudentRecord(
        roll_number=roll_number,
        name=name,
        individual_embeddings=individual_embeddings,
        representative_embedding=representative,
    )

    store = EmbeddingStore.load_embeddings(settings.embeddings_file)
    if roll_number in store:
        EmbeddingStore.update_student(record, settings.embeddings_file)
    else:
        EmbeddingStore.add_student(record, settings.embeddings_file)

    logger.info(
        "EmbeddingStore updated for '%s' (%s) — %d accepted embeddings",
        name,
        roll_number,
        accepted_count,
    )

    # Reload the recognizer singleton so the new embeddings take effect
    try:
        from backend.services import recognition_service  # noqa: PLC0415

        recognition_service.reload_embeddings()
        logger.info("Recognizer reloaded after embedding update for '%s'", roll_number)
    except Exception as exc:
        logger.warning(
            "Could not reload recognizer after embedding update for '%s': %s",
            roll_number,
            exc,
        )

    if accepted_count < 10:
        warning = (
            f"Only {accepted_count} image(s) passed quality filtering. "
            "Recognition accuracy may be reduced. "
            "Capture at least 10 images for best results."
        )

    return {"accepted_count": accepted_count, "warning": warning}


# ---------------------------------------------------------------------------
# 9. delete_images
# ---------------------------------------------------------------------------

async def delete_images(roll_number: str, db) -> int:
    """
    Delete all ``.jpg`` images in the student's dataset folder.

    The folder itself is preserved (not removed). Returns the number of files
    deleted.

    Parameters
    ----------
    roll_number : str
    db : aiosqlite.Connection

    Returns
    -------
    int
        Number of image files deleted.

    Raises
    ------
    KeyError
        If the student does not exist.

    Requirements: 4.5
    """
    student = await get_student(roll_number, db)
    name = student["name"]
    folder = _student_folder(name, roll_number)

    if not os.path.isdir(folder):
        logger.info("delete_images: folder not found for '%s', nothing to delete.", roll_number)
        return 0

    jpg_files = [
        f for f in os.listdir(folder) if f.lower().endswith(".jpg")
    ]
    deleted = 0
    for filename in jpg_files:
        filepath = os.path.join(folder, filename)
        try:
            os.remove(filepath)
            deleted += 1
        except OSError as exc:
            logger.warning("Could not delete image %s: %s", filepath, exc)

    logger.info("Deleted %d image(s) from folder '%s'", deleted, folder)
    return deleted
