"""
face_matcher.py

Cosine-similarity identity matching for the AI Attendance Monitoring System.

This module provides:
  - RecognitionResult: a typed dataclass representing the outcome of a single
    face-match query, importable by backend route handlers and recognizer.py.
  - FaceMatcher: compares a query embedding against the in-memory embedding
    store and returns the best-matching RecognitionResult.


"""

import dataclasses
import logging
from typing import Dict, Tuple

import numpy as np

try:
    from .config import RECOGNITION_THRESHOLD
    from .embedding_generator import StudentRecord
except ImportError:  # Allow running/importing this module directly as a script
    from config import RECOGNITION_THRESHOLD
    from embedding_generator import StudentRecord


# Module-level 

logger = logging.getLogger("face_matcher")



# RecognitionResult dataclass — 


@dataclasses.dataclass
class RecognitionResult:
    """
    Structured result for a single face-match query.

    Fields:
        name (str):               Student name, or "Unknown" when unrecognised.
        roll_number (str):        Student roll number, or "" when unrecognised.
        confidence_score (float): Raw cosine similarity value (always present).
        recognition_status (str): "Known" or "Unknown".
        bounding_box (Tuple):     (x, y, width, height) in pixels.
    """

    name: str
    roll_number: str
    confidence_score: float
    recognition_status: str
    bounding_box: Tuple[int, int, int, int]



# FaceMatcher class — 


class FaceMatcher:
    """
    Compares a query embedding against all stored student embeddings using
    cosine similarity (dot product of L2-normalised vectors) and returns a
    RecognitionResult indicating the best match or Unknown.

    Args:
        threshold (float): Minimum cosine similarity required to classify a
            face as Known. Defaults to RECOGNITION_THRESHOLD from config.py.
    """

    def __init__(self, threshold: float = RECOGNITION_THRESHOLD) -> None:
        self._threshold = threshold
        logger.debug(
            "FaceMatcher initialised with threshold=%.4f", self._threshold
        )

    def match(
        self,
        query_embedding: np.ndarray,
        store: Dict[str, StudentRecord],
        bounding_box: Tuple[int, int, int, int],
    ) -> RecognitionResult:
        """
        Find the best-matching student for the given query embedding.

        Algorithm:
          1. Filter out records whose representative_embedding is None (log DEBUG).
          2. If no valid records remain → Unknown result with confidence 0.0.
          3. Compute score = np.dot(query_embedding, record.representative_embedding)
             for every valid record (dot product of unit vectors == cosine similarity).
          4. Select the record with the maximum score.
          5. If score < threshold → Unknown; otherwise → Known.
          6. Always populate confidence_score and bounding_box.

        Args:
            query_embedding: L2-normalised float32 array of shape (512,).
            store: Mapping of roll_number → StudentRecord.
            bounding_box: (x, y, width, height) tuple from the face detector.

        Returns:
            RecognitionResult with all five fields populated.
        """
        # Step 1 — filter records with no representative embedding
        valid_records = {}
        for roll_number, record in store.items():
            if record.representative_embedding is None:
                logger.debug(
                    "Skipping record '%s' (%s): representative_embedding is None",
                    roll_number,
                    record.name,
                )
            else:
                valid_records[roll_number] = record

        # Step 2 — empty store: return Unknown with confidence 0.0
        if not valid_records:
            logger.debug(
                "No valid records in store; returning Unknown (confidence=0.0)"
            )
            return RecognitionResult(
                name="Unknown",
                roll_number="",
                confidence_score=0.0,
                recognition_status="Unknown",
                bounding_box=bounding_box,
            )

        # Steps 3 & 4 — compute dot-product similarities and find best match
        best_roll: str = ""
        best_score: float = -float("inf")

        for roll_number, record in valid_records.items():
            score = float(np.dot(query_embedding, record.representative_embedding))
            if score > best_score:
                best_score = score
                best_roll = roll_number

        best_record = valid_records[best_roll]

        # Steps 5 & 6 — threshold classification
        if best_score < self._threshold:
            return RecognitionResult(
                name="Unknown",
                roll_number="",
                confidence_score=best_score,
                recognition_status="Unknown",
                bounding_box=bounding_box,
            )

        return RecognitionResult(
            name=best_record.name,
            roll_number=best_record.roll_number,
            confidence_score=best_score,
            recognition_status="Known",
            bounding_box=bounding_box,
        )
