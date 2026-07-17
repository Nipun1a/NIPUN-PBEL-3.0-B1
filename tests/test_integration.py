"""
tests/test_integration.py

End-to-end integration tests for the face-embedding-generation pipeline.

These tests load the REAL face recognition model (InsightFace ArcFace or
FaceNet fallback) and run against the actual images in CollectedImages/.
They are intentionally slow (30–120 seconds each) and are guarded behind
the ``integration`` pytest mark.

Run all tests:
    pytest tests/test_integration.py -v

Run only unit tests (skip integration):
    pytest -m "not integration"

Run only integration tests:
    pytest -m integration
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path so imports work regardless of how
# pytest is invoked (from repo root, from tests/, etc.)
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

# ---------------------------------------------------------------------------
# Skip the entire module if neither face recognition backend is installed
# ---------------------------------------------------------------------------
try:
    import insightface  # noqa: F401
    _BACKEND_AVAILABLE = True
except ImportError:
    try:
        import facenet_pytorch  # noqa: F401
        _BACKEND_AVAILABLE = True
    except ImportError:
        _BACKEND_AVAILABLE = False

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Guarded imports: only fail with a helpful message at collection time
# ---------------------------------------------------------------------------
from embedding_generator import (
    generate_and_save_all,
    add_new_student,
    update_existing_student,
    remove_student,
    EmbeddingStore,
    EMBEDDING_DIM,
)
from models import StudentDetails

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ROOT = r"c:\Users\Nipun\PBEL\CollectedImages"

# Module-level mark: every test in this file requires the integration mark
pytestmark = pytest.mark.integration



# Helper: skip when no model backend is available

def _require_backend():
    if not _BACKEND_AVAILABLE:
        pytest.skip(
            "No face recognition backend installed. "
            "Install one of: 'pip install insightface onnxruntime' "
            "or 'pip install facenet-pytorch'"
        )



# — Full dataset bootstrap integration test


@pytest.mark.integration
def test_generate_and_save_all(tmp_path):
    """
    Call generate_and_save_all against the real CollectedImages directory and
    verify the persisted store contains correct, normalised embeddings for both
    students (Nipun / roll 101, Yug / roll 102).

    Validates: Requirements 4.4, 8.2, 11.1, 11.3
    """
    _require_backend()

    output_filepath = str(tmp_path / "embeddings.pkl")

    
    # 1. Run the full bootstrap
    
    num_saved = generate_and_save_all(
        dataset_root=DATASET_ROOT,
        output_filepath=output_filepath,
    )

   
    # 2. Assert count of saved students
    
    assert num_saved == 2, (
        f"Expected 2 students to be saved, got {num_saved}"
    )

    
    # 3. Load via EmbeddingStore and assert structure
     
    store = EmbeddingStore.load_embeddings(output_filepath)

    assert len(store) == 2, (
        f"Expected exactly 2 records in store, got {len(store)}: {list(store.keys())}"
    )
    assert "101" in store, "Expected roll_number '101' (Nipun) in store"
    assert "102" in store, "Expected roll_number '102' (Yug) in store"

  
    # 4. Per-record assertions
   
    for roll_number in ("101", "102"):
        record = store[roll_number]

        # individual_embeddings must be non-empty and at most 100
        assert len(record.individual_embeddings) > 0, (
            f"Roll {roll_number}: individual_embeddings must not be empty"
        )
        assert len(record.individual_embeddings) <= 100, (
            f"Roll {roll_number}: individual_embeddings length {len(record.individual_embeddings)} "
            f"exceeds 100"
        )

        rep = record.representative_embedding
        assert rep is not None, (
            f"Roll {roll_number}: representative_embedding must not be None"
        )

        # dtype must be float32
        assert rep.dtype == np.float32, (
            f"Roll {roll_number}: representative_embedding.dtype expected float32, "
            f"got {rep.dtype}"
        )

        # shape must be (EMBEDDING_DIM,)
        assert rep.shape == (EMBEDDING_DIM,), (
            f"Roll {roll_number}: representative_embedding.shape expected "
            f"({EMBEDDING_DIM},), got {rep.shape}"
        )

        # L2 norm must be within 1e-6 of 1.0
        norm = float(np.linalg.norm(rep))
        assert abs(norm - 1.0) < 1e-6, (
            f"Roll {roll_number}: representative_embedding L2 norm {norm:.8f} "
            f"is not within 1e-6 of 1.0"
        )


# Task 10.2 — Incremental CRUD lifecycle integration test


@pytest.mark.integration
def test_crud_lifecycle(tmp_path):
    """
    Starting from an empty store, exercise the full add → update → remove
    lifecycle for a single student using the real model and real images.

    Each step is run in sequence within this single test function so that
    failures are easy to attribute to a specific CRUD operation.

    
    """
    _require_backend()

    crud_store_path = str(tmp_path / "crud_store.pkl")

    student_details = StudentDetails(name="Nipun", roll_number="101")

    
    # Step 1 — add_new_student
    # The store is empty so roll "101" does not exist yet.
    
    result = add_new_student(
        student_details,
        dataset_root=DATASET_ROOT,
        output_filepath=crud_store_path,
    )

    assert result.success is True, (
        f"add_new_student should succeed for a fresh store, "
        f"got success=False with message: {result.message!r}"
    )

    fetched = EmbeddingStore.get_student("101", crud_store_path)
    assert fetched is not None, (
        "get_student('101') should return a record after add_new_student"
    )

    
    # Step 2 — update_existing_student
    # Re-generate and overwrite the record for roll "101".
    
    update_result = update_existing_student(
        "101",
        dataset_root=DATASET_ROOT,
        output_filepath=crud_store_path,
    )

    assert update_result.success is True, (
        f"update_existing_student should succeed for existing roll '101', "
        f"got success=False with message: {update_result.message!r}"
    )

    # Record should still be retrievable after update
    updated_record = EmbeddingStore.get_student("101", crud_store_path)
    assert updated_record is not None, (
        "get_student('101') should still return a record after update_existing_student"
    )

    
    # Step 3 — remove_student

    remove_result = remove_student(
        "101",
        output_filepath=crud_store_path,
    )

    assert remove_result.success is True, (
        f"remove_student should succeed for existing roll '101', "
        f"got success=False with message: {remove_result.message!r}"
    )

    # Step 4 — verify removal
    
    gone = EmbeddingStore.get_student("101", crud_store_path)
    assert gone is None, (
        f"get_student('101') should return None after remove_student, "
        f"got {gone!r}"
    )
