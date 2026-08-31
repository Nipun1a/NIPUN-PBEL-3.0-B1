"""
unknown_faces_service.py

Async service layer for the Unknown Face Gallery feature.

All SQL queries use parameterised placeholders (``?``) exclusively —
no string interpolation of user-supplied values.

Functions:
    get_paginated      — filtered, paginated list with base64 image_data
    get_by_id          — single record by id
    delete_one         — delete DB row + JPEG crop file
    bulk_delete        — delete multiple rows + files, return count
    register_from_unknown — create student from unknown face crop + generate embeddings
    get_stats          — aggregated counts and average confidence

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7
"""
from __future__ import annotations

import base64
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Ensure the project root (one level above backend/) is on sys.path so that
# the pre-existing ML modules (EmbeddingGenerator, EmbeddingStore) are
# importable without modification.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — read a JPEG file and base64-encode it
# ---------------------------------------------------------------------------

def _read_image_base64(image_path: str) -> str:
    """
    Read the JPEG at *image_path* and return its base64-encoded bytes as a
    UTF-8 string.  Returns an empty string if *image_path* is falsy or the
    file does not exist.
    """
    if not image_path:
        return ""
    if not os.path.exists(image_path):
        return ""
    try:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except OSError as exc:
        logger.warning("Could not read image file %s: %s", image_path, exc)
        return ""


def _delete_file_if_exists(image_path: str) -> None:
    """Remove a file from disk if the path is non-empty and the file exists."""
    if not image_path:
        return
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            logger.info("Deleted unknown face crop: %s", image_path)
        except OSError as exc:
            logger.warning("Could not delete file %s: %s", image_path, exc)


# ---------------------------------------------------------------------------
# 1. get_paginated
# ---------------------------------------------------------------------------

async def get_paginated(
    db,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: Optional[float] = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Return a paginated list of unknown face records with optional filters.

    Each record includes ``image_data``: a base64-encoded JPEG read from the
    ``image_path`` column, or an empty string if the file does not exist.

    Parameters
    ----------
    db : aiosqlite.Connection
    date : str or None
        Exact calendar date filter (``YYYY-MM-DD``).
    start_date : str or None
        Inclusive start of a date range (``YYYY-MM-DD``).
    end_date : str or None
        Inclusive end of a date range (``YYYY-MM-DD``).
    min_confidence : float or None
        Minimum ``confidence_score`` filter.
    page : int
        1-indexed page number (default 1).
    page_size : int
        Records per page (default 20, max 100 enforced by router).

    Returns
    -------
    dict
        ``{ total, page, page_size, records }``

    Requirements: 21.1
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 100))

    conditions: list[str] = []
    params: list[Any] = []

    if date is not None:
        conditions.append("DATE(timestamp) = ?")
        params.append(date)
    if start_date is not None:
        conditions.append("DATE(timestamp) >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("DATE(timestamp) <= ?")
        params.append(end_date)
    if min_confidence is not None:
        conditions.append("confidence_score >= ?")
        params.append(min_confidence)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Total count
    count_sql = f"SELECT COUNT(*) FROM unknown_faces {where_clause}"  # noqa: S608
    async with db.execute(count_sql, params) as cursor:
        row = await cursor.fetchone()
    total: int = row[0] if row else 0

    # Paginated records
    offset = (page - 1) * page_size
    data_sql = (
        f"SELECT id, timestamp, confidence_score, image_path, created_at "  # noqa: S608
        f"FROM unknown_faces {where_clause} "
        f"ORDER BY timestamp DESC "
        f"LIMIT ? OFFSET ?"
    )
    data_params = params + [page_size, offset]

    async with db.execute(data_sql, data_params) as cursor:
        rows = await cursor.fetchall()

    records = []
    for r in rows:
        image_path = r["image_path"] or ""
        records.append(
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "confidence_score": r["confidence_score"],
                "image_data": _read_image_base64(image_path),
                "image_path": image_path,
                "created_at": r["created_at"],
            }
        )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": records,
    }


# ---------------------------------------------------------------------------
# 2. get_by_id
# ---------------------------------------------------------------------------

async def get_by_id(id: int, db) -> dict:
    """
    Fetch a single unknown face record by its primary key.

    Returns
    -------
    dict
        Record dict including ``image_data`` (base64 JPEG or empty string).

    Raises
    ------
    KeyError
        If no record with *id* exists.

    Requirements: 21.2
    """
    async with db.execute(
        "SELECT id, timestamp, confidence_score, image_path, created_at "
        "FROM unknown_faces WHERE id = ?",
        (id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise KeyError(f"Unknown face record with id {id} not found.")

    image_path = row["image_path"] or ""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "confidence_score": row["confidence_score"],
        "image_data": _read_image_base64(image_path),
        "image_path": image_path,
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# 3. delete_one
# ---------------------------------------------------------------------------

async def delete_one(id: int, db) -> None:
    """
    Delete an unknown face record from the database and its JPEG crop from disk.

    Parameters
    ----------
    id : int
        Primary key of the record to delete.
    db : aiosqlite.Connection

    Raises
    ------
    KeyError
        If no record with *id* exists.

    Requirements: 21.3
    """
    # Fetch first so we can delete the file and raise KeyError if not found.
    async with db.execute(
        "SELECT id, image_path FROM unknown_faces WHERE id = ?",
        (id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise KeyError(f"Unknown face record with id {id} not found.")

    image_path: str = row["image_path"] or ""

    # Delete the database row.
    await db.execute("DELETE FROM unknown_faces WHERE id = ?", (id,))
    await db.commit()

    # Delete the JPEG crop from disk (no-op if path is empty or file missing).
    _delete_file_if_exists(image_path)


# ---------------------------------------------------------------------------
# 4. bulk_delete
# ---------------------------------------------------------------------------

async def bulk_delete(ids: list[int], db) -> int:
    """
    Delete multiple unknown face records and their associated JPEG crop files.

    Records that do not exist are silently skipped (no KeyError raised).

    Parameters
    ----------
    ids : list[int]
        Primary keys to delete.
    db : aiosqlite.Connection

    Returns
    -------
    int
        Number of rows actually deleted.

    Requirements: 21.4
    """
    if not ids:
        return 0

    # Fetch image_paths for existing records before deleting.
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT id, image_path FROM unknown_faces WHERE id IN ({placeholders})",  # noqa: S608
        ids,
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        return 0

    # Collect image paths before deletion.
    image_paths = [r["image_path"] or "" for r in rows]
    existing_ids = [r["id"] for r in rows]

    # Delete the rows.
    id_placeholders = ",".join("?" * len(existing_ids))
    async with db.execute(
        f"DELETE FROM unknown_faces WHERE id IN ({id_placeholders})",  # noqa: S608
        existing_ids,
    ) as cursor:
        deleted_count = cursor.rowcount

    await db.commit()

    # Delete the JPEG files from disk.
    for path in image_paths:
        _delete_file_if_exists(path)

    return deleted_count


# ---------------------------------------------------------------------------
# 5. register_from_unknown
# ---------------------------------------------------------------------------

async def register_from_unknown(id: int, student_data: dict, db) -> dict:
    """
    Register a new student using an unknown face crop as the first training image.

    Steps
    -----
    1. Fetch the unknown face record (raise ``KeyError`` if not found).
    2. Check that ``roll_number`` does not already exist in ``students``
       (raise ``ValueError`` with "409" prefix if it does).
    3. Create a new ``students`` row in the database.
    4. Copy / save the JPEG crop to
       ``CollectedImages/{Name}_{RollNumber}/{Name}_001.jpg``.
    5. Run ``EmbeddingGenerator`` on that folder to generate embeddings.
    6. Call ``EmbeddingStore.update_student()`` (or ``add_student`` if new)
       to persist the embedding.
    7. Call ``recognition_service.reload_embeddings()`` to hot-reload.
    8. Return ``{ "student": student_dict, "warning": warning_message }``.

    Parameters
    ----------
    id : int
        Primary key of the unknown face record to register from.
    student_data : dict
        Must contain ``roll_number`` and ``name``; may contain
        ``department``, ``email``, ``phone``.
    db : aiosqlite.Connection

    Returns
    -------
    dict
        ``{ "student": <student_dict>, "warning": <str> }``

    Raises
    ------
    KeyError
        If no unknown face record with *id* exists.
    ValueError
        If ``roll_number`` already exists (message contains "409").

    Requirements: 21.5, 21.6
    """
    # ------------------------------------------------------------------
    # 1. Fetch unknown face record
    # ------------------------------------------------------------------
    async with db.execute(
        "SELECT id, image_path FROM unknown_faces WHERE id = ?",
        (id,),
    ) as cursor:
        uf_row = await cursor.fetchone()

    if uf_row is None:
        raise KeyError(f"Unknown face record with id {id} not found.")

    image_path: str = uf_row["image_path"] or ""

    # ------------------------------------------------------------------
    # 2. Check for duplicate roll_number
    # ------------------------------------------------------------------
    roll_number: str = student_data["roll_number"]
    name: str = student_data["name"]
    department: str = student_data.get("department", "")
    email: str = student_data.get("email", "")
    phone: str = student_data.get("phone", "")

    async with db.execute(
        "SELECT roll_number FROM students WHERE roll_number = ?",
        (roll_number,),
    ) as cursor:
        existing = await cursor.fetchone()

    if existing is not None:
        raise ValueError(
            f"409: Student with roll number {roll_number} already exists"
        )

    # ------------------------------------------------------------------
    # 3. Create student row
    # ------------------------------------------------------------------
    now = datetime.utcnow().isoformat()
    await db.execute(
        """
        INSERT INTO students (roll_number, name, department, email, phone, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (roll_number, name, department, email, phone, now, now),
    )
    await db.commit()

    student_dict = {
        "roll_number": roll_number,
        "name": name,
        "department": department,
        "email": email,
        "phone": phone,
        "created_at": now,
        "updated_at": now,
    }

    # ------------------------------------------------------------------
    # 4. Save crop JPEG to CollectedImages/{Name}_{RollNumber}/{Name}_001.jpg
    # ------------------------------------------------------------------
    folder_name = f"{name}_{roll_number}"
    dest_folder = os.path.join(settings.dataset_root, folder_name)
    os.makedirs(dest_folder, exist_ok=True)

    dest_image_path = os.path.join(dest_folder, f"{name}_001.jpg")

    if image_path and os.path.exists(image_path):
        import shutil
        shutil.copy2(image_path, dest_image_path)
        logger.info("Copied crop %s → %s", image_path, dest_image_path)
    else:
        logger.warning(
            "Unknown face crop not found at '%s'; student folder created but empty.",
            image_path,
        )

    # ------------------------------------------------------------------
    # 5. Generate embeddings for the folder
    # ------------------------------------------------------------------
    warning = ""

    try:
        from embedding_generator import EmbeddingGenerator, EmbeddingStore, StudentRecord  # noqa: PLC0415

        generator = EmbeddingGenerator()
        individual_embeddings = generator.generate_embeddings(dest_folder)
        accepted_count = len(individual_embeddings)

        if accepted_count == 0:
            warning = (
                "No images passed quality filtering. "
                "Embedding store was not updated. "
                "Please capture more images for this student."
            )
            logger.warning(
                "register_from_unknown: no accepted images for roll_number '%s'",
                roll_number,
            )
        else:
            # Aggregate into a representative embedding
            representative = generator._aggregate_embeddings(
                individual_embeddings, name, roll_number
            )

            if representative is None:
                warning = (
                    "Aggregated embedding produced a zero-norm vector. "
                    "Embedding store was not updated."
                )
                logger.error(
                    "register_from_unknown: zero-norm representative for '%s'",
                    roll_number,
                )
            else:
                # ----------------------------------------------------------
                # 6. Persist to EmbeddingStore
                # ----------------------------------------------------------
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
                    "EmbeddingStore updated for '%s' (%s) — %d accepted image(s)",
                    name,
                    roll_number,
                    accepted_count,
                )

                if accepted_count < 10:
                    warning = (
                        f"Only {accepted_count} image(s) passed quality filtering. "
                        "Recognition accuracy may be reduced. "
                        "Capture at least 10 images for best results."
                    )

    except Exception as exc:
        logger.error(
            "register_from_unknown: embedding generation failed for '%s': %s",
            roll_number,
            exc,
        )
        if not warning:
            warning = f"Embedding generation failed: {exc}"

    # ------------------------------------------------------------------
    # 7. Hot-reload the recognizer singleton
    # ------------------------------------------------------------------
    try:
        from backend.services import recognition_service  # noqa: PLC0415

        recognition_service.reload_embeddings()
        logger.info(
            "Recognizer reloaded after registering unknown face as '%s'", roll_number
        )
    except Exception as exc:
        logger.warning(
            "Could not reload recognizer after registering '%s': %s", roll_number, exc
        )

    return {"student": student_dict, "warning": warning}


# ---------------------------------------------------------------------------
# 6. get_stats
# ---------------------------------------------------------------------------

async def get_stats(db) -> dict:
    """
    Return aggregated statistics for the unknown faces table.

    Returns
    -------
    dict
        ``{ total_logged, logged_today, logged_this_week, average_confidence_score }``

    Requirements: 21.7
    """
    now = datetime.utcnow()
    today_str = now.date().isoformat()

    # Start of the current week (Monday)
    week_start = (now - timedelta(days=now.weekday())).date().isoformat()

    async with db.execute("SELECT COUNT(*) FROM unknown_faces") as cursor:
        row = await cursor.fetchone()
    total_logged: int = row[0] if row else 0

    async with db.execute(
        "SELECT COUNT(*) FROM unknown_faces WHERE DATE(timestamp) = ?",
        (today_str,),
    ) as cursor:
        row = await cursor.fetchone()
    logged_today: int = row[0] if row else 0

    async with db.execute(
        "SELECT COUNT(*) FROM unknown_faces WHERE DATE(timestamp) >= ?",
        (week_start,),
    ) as cursor:
        row = await cursor.fetchone()
    logged_this_week: int = row[0] if row else 0

    async with db.execute(
        "SELECT AVG(confidence_score) FROM unknown_faces"
    ) as cursor:
        row = await cursor.fetchone()
    avg_raw = row[0] if row else None
    average_confidence_score: float = round(float(avg_raw), 4) if avg_raw is not None else 0.0

    return {
        "total_logged": total_logged,
        "logged_today": logged_today,
        "logged_this_week": logged_this_week,
        "average_confidence_score": average_confidence_score,
    }
