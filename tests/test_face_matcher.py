"""
tests/test_face_matcher.py

Unit tests for FaceMatcher and RecognitionResult in face_matcher.py.

Covers:
  - RecognitionResult dataclass instantiation and field presence
  - FaceMatcher.__init__ with default and custom threshold
  - FaceMatcher.match() — empty store
  - FaceMatcher.match() — valid store, score >= threshold (Known)
  - FaceMatcher.match() — valid store, score < threshold (Unknown)
  - FaceMatcher.match() — store where all embeddings are None (Unknown)
  - Result always has bounding_box populated
"""

import sys
import os

# Ensure the project root is on the path so imports resolve without a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from face_matcher import FaceMatcher, RecognitionResult
from embedding_generator import StudentRecord
from config import RECOGNITION_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector of length `dim`."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_record(
    roll_number: str,
    name: str,
    embedding: np.ndarray | None,
) -> StudentRecord:
    return StudentRecord(
        roll_number=roll_number,
        name=name,
        individual_embeddings=[embedding] if embedding is not None else [],
        representative_embedding=embedding,
    )


DUMMY_BOX = (10, 20, 100, 80)  # (x, y, width, height)


# ---------------------------------------------------------------------------
# RecognitionResult tests
# ---------------------------------------------------------------------------

class TestRecognitionResult:
    def test_instantiation_with_all_fields(self):
        result = RecognitionResult(
            name="Alice",
            roll_number="101",
            confidence_score=0.85,
            recognition_status="Known",
            bounding_box=DUMMY_BOX,
        )
        assert result.name == "Alice"
        assert result.roll_number == "101"
        assert result.confidence_score == pytest.approx(0.85)
        assert result.recognition_status == "Known"
        assert result.bounding_box == DUMMY_BOX

    def test_has_all_five_required_fields(self):
        """All five required fields must be present on every instance."""
        result = RecognitionResult(
            name="Bob",
            roll_number="202",
            confidence_score=0.0,
            recognition_status="Unknown",
            bounding_box=(0, 0, 50, 50),
        )
        for field in ("name", "roll_number", "confidence_score",
                      "recognition_status", "bounding_box"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_unknown_result_fields(self):
        result = RecognitionResult(
            name="Unknown",
            roll_number="",
            confidence_score=0.0,
            recognition_status="Unknown",
            bounding_box=DUMMY_BOX,
        )
        assert result.name == "Unknown"
        assert result.roll_number == ""
        assert result.recognition_status == "Unknown"


# ---------------------------------------------------------------------------
# FaceMatcher.__init__ tests
# ---------------------------------------------------------------------------

class TestFaceMatcherInit:
    def test_default_threshold(self):
        matcher = FaceMatcher()
        assert matcher._threshold == pytest.approx(RECOGNITION_THRESHOLD)

    def test_custom_threshold(self):
        matcher = FaceMatcher(threshold=0.75)
        assert matcher._threshold == pytest.approx(0.75)

    def test_threshold_zero(self):
        matcher = FaceMatcher(threshold=0.0)
        assert matcher._threshold == pytest.approx(0.0)

    def test_threshold_one(self):
        matcher = FaceMatcher(threshold=1.0)
        assert matcher._threshold == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FaceMatcher.match() — empty store
# ---------------------------------------------------------------------------

class TestFaceMatcherEmptyStore:
    def test_empty_store_returns_unknown(self):
        matcher = FaceMatcher()
        q = _unit_vec()
        result = matcher.match(q, {}, DUMMY_BOX)
        assert result.recognition_status == "Unknown"
        assert result.name == "Unknown"
        assert result.roll_number == ""

    def test_empty_store_confidence_is_zero(self):
        matcher = FaceMatcher()
        result = matcher.match(_unit_vec(), {}, DUMMY_BOX)
        assert result.confidence_score == pytest.approx(0.0)

    def test_empty_store_bounding_box_populated(self):
        matcher = FaceMatcher()
        result = matcher.match(_unit_vec(), {}, DUMMY_BOX)
        assert result.bounding_box == DUMMY_BOX


# ---------------------------------------------------------------------------
# FaceMatcher.match() — all embeddings are None
# ---------------------------------------------------------------------------

class TestFaceMatcherAllNoneEmbeddings:
    def test_all_none_returns_unknown(self):
        store = {
            "101": _make_record("101", "Alice", None),
            "102": _make_record("102", "Bob", None),
        }
        matcher = FaceMatcher()
        result = matcher.match(_unit_vec(), store, DUMMY_BOX)
        assert result.recognition_status == "Unknown"
        assert result.name == "Unknown"
        assert result.roll_number == ""

    def test_all_none_confidence_is_zero(self):
        store = {"101": _make_record("101", "Alice", None)}
        matcher = FaceMatcher()
        result = matcher.match(_unit_vec(), store, DUMMY_BOX)
        assert result.confidence_score == pytest.approx(0.0)

    def test_all_none_bounding_box_populated(self):
        store = {"101": _make_record("101", "Alice", None)}
        matcher = FaceMatcher()
        result = matcher.match(_unit_vec(), store, DUMMY_BOX)
        assert result.bounding_box == DUMMY_BOX


# ---------------------------------------------------------------------------
# FaceMatcher.match() — score >= threshold  →  Known
# ---------------------------------------------------------------------------

class TestFaceMatcherKnownResult:
    def _make_known_store_and_query(self, threshold: float = 0.5):
        """
        Build a query and a store where the best score is guaranteed >= threshold.
        Using the same vector for query and record gives score ~= 1.0 (max possible).
        """
        q = _unit_vec(seed=42)
        record = _make_record("101", "Alice", q.copy())
        store = {"101": record}
        return q, store

    def test_known_status_returned(self):
        matcher = FaceMatcher(threshold=0.5)
        q, store = self._make_known_store_and_query()
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.recognition_status == "Known"

    def test_known_name_and_roll_number(self):
        matcher = FaceMatcher(threshold=0.5)
        q, store = self._make_known_store_and_query()
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.name == "Alice"
        assert result.roll_number == "101"

    def test_known_confidence_equals_dot_product(self):
        matcher = FaceMatcher(threshold=0.5)
        q = _unit_vec(seed=42)
        emb = _unit_vec(seed=7)
        store = {"101": _make_record("101", "Alice", emb)}
        result = matcher.match(q, store, DUMMY_BOX)
        expected_score = float(np.dot(q, emb))
        assert result.confidence_score == pytest.approx(expected_score, abs=1e-5)

    def test_known_bounding_box_populated(self):
        matcher = FaceMatcher(threshold=0.5)
        q, store = self._make_known_store_and_query()
        box = (5, 15, 200, 150)
        result = matcher.match(q, store, box)
        assert result.bounding_box == box

    def test_selects_highest_similarity_record(self):
        """When multiple records exist, the one with highest dot product wins."""
        q = _unit_vec(seed=1)
        # Create two records; make one identical to the query (score ~= 1.0)
        close = q.copy()
        far = _unit_vec(seed=99)
        store = {
            "101": _make_record("101", "Close", close),
            "102": _make_record("102", "Far", far),
        }
        matcher = FaceMatcher(threshold=0.5)
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.roll_number == "101"
        assert result.name == "Close"


# ---------------------------------------------------------------------------
# FaceMatcher.match() — score < threshold  →  Unknown
# ---------------------------------------------------------------------------

class TestFaceMatcherUnknownResult:
    def _make_below_threshold_store_and_query(self):
        """
        Build a query and a store guaranteed to produce a score < threshold.
        We use a very high threshold (1.1 > max cosine similarity of 1.0).
        """
        q = _unit_vec(seed=1)
        emb = _unit_vec(seed=2)
        store = {"101": _make_record("101", "Bob", emb)}
        # Force threshold above any possible cosine similarity
        matcher = FaceMatcher(threshold=1.1)
        return matcher, q, store

    def test_below_threshold_returns_unknown_status(self):
        matcher, q, store = self._make_below_threshold_store_and_query()
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.recognition_status == "Unknown"

    def test_below_threshold_name_is_unknown(self):
        matcher, q, store = self._make_below_threshold_store_and_query()
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.name == "Unknown"

    def test_below_threshold_roll_number_is_empty(self):
        matcher, q, store = self._make_below_threshold_store_and_query()
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.roll_number == ""

    def test_below_threshold_confidence_is_best_score(self):
        """confidence_score must equal the best (but sub-threshold) dot product."""
        q = _unit_vec(seed=3)
        emb = _unit_vec(seed=4)
        expected_score = float(np.dot(q, emb))
        store = {"101": _make_record("101", "Charlie", emb)}
        matcher = FaceMatcher(threshold=1.1)
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.confidence_score == pytest.approx(expected_score, abs=1e-5)

    def test_below_threshold_bounding_box_populated(self):
        matcher, q, store = self._make_below_threshold_store_and_query()
        box = (0, 0, 300, 200)
        result = matcher.match(q, store, box)
        assert result.bounding_box == box


# ---------------------------------------------------------------------------
# Mixed store: some None, some valid — None records must be skipped
# ---------------------------------------------------------------------------

class TestFaceMatcherMixedStore:
    def test_none_records_skipped_known_selected(self):
        """Only the valid record should be eligible; None records are ignored."""
        q = _unit_vec(seed=5)
        valid_emb = q.copy()  # identical → score ~= 1.0
        store = {
            "101": _make_record("101", "Valid", valid_emb),
            "102": _make_record("102", "NullStudent", None),
        }
        matcher = FaceMatcher(threshold=0.5)
        result = matcher.match(q, store, DUMMY_BOX)
        assert result.recognition_status == "Known"
        assert result.roll_number == "101"

    def test_none_records_skipped_check_unknown_not_selected(self):
        """None-embedding records must never appear in the result."""
        q = _unit_vec(seed=6)
        valid_emb = _unit_vec(seed=7)
        store = {
            "101": _make_record("101", "Valid", valid_emb),
            "102": _make_record("102", "NullStudent", None),
        }
        matcher = FaceMatcher(threshold=1.1)  # below threshold
        result = matcher.match(q, store, DUMMY_BOX)
        # Should be Unknown from the valid record, not the None record
        assert result.roll_number != "102"
        assert result.name != "NullStudent"


# ---------------------------------------------------------------------------
# Bounding box always populated — edge cases
# ---------------------------------------------------------------------------

class TestBoundingBoxAlwaysPopulated:
    def test_zero_box_propagated_empty_store(self):
        matcher = FaceMatcher()
        box = (0, 0, 0, 0)
        result = matcher.match(_unit_vec(), {}, box)
        assert result.bounding_box == box

    def test_large_box_propagated_known(self):
        q = _unit_vec(seed=10)
        store = {"101": _make_record("101", "Dana", q.copy())}
        matcher = FaceMatcher(threshold=0.5)
        box = (1000, 2000, 300, 400)
        result = matcher.match(q, store, box)
        assert result.bounding_box == box

    def test_large_box_propagated_unknown(self):
        q = _unit_vec(seed=11)
        emb = _unit_vec(seed=12)
        store = {"101": _make_record("101", "Eve", emb)}
        matcher = FaceMatcher(threshold=1.1)
        box = (50, 60, 70, 80)
        result = matcher.match(q, store, box)
        assert result.bounding_box == box
