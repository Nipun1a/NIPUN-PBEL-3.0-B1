"""
tests/test_embedding_generator.py

Unit tests for EmbeddingGenerator and EmbeddingStore.
No real model weights are loaded — all model calls are mocked.


"""

import os
import sys
import pickle
import tempfile
import dataclasses

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — ensure workspace root is importable regardless of how pytest
# is invoked (e.g. from tests/ or from project root).

_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from embedding_generator import (
    EmbeddingGenerator,
    EmbeddingStore,
    StudentRecord,
    EMBEDDING_DIM,
)
from face_detector import FaceDetection



# Helper — create an EmbeddingGenerator without triggering __init__ (no model
# download, no FaceDetector construction).
# ===========================================================================

def make_mock_generator(mock_embedding: np.ndarray = None) -> EmbeddingGenerator:
    """
    Bypass __init__ entirely via __new__ and wire up minimal state.

    Parameters
    ----------
    mock_embedding : np.ndarray, optional
        If provided, configure gen._model.get() to return a mock face whose
        .embedding attribute is this array (InsightFace code-path).
    """
    gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
    gen._face_size = (160, 160)
    gen._min_quality_score = 60.0
    gen._min_face_size = 60
    gen._use_insightface = True
    gen._model = MagicMock()
    gen._face_detector = MagicMock()

    # Ensure the code uses the app.get() fallback path, not rec_model.get_feat().
    # models.get("recognition") returning None triggers the fallback branch.
    gen._model.models.get.return_value = None

    if mock_embedding is not None:
        mock_face = MagicMock()
        mock_face.embedding = mock_embedding
        gen._model.get.return_value = [mock_face]

    return gen


def _make_sharp_bgr(h: int = 100, w: int = 100) -> np.ndarray:
    """
    Return a synthetic BGR image with enough high-frequency content that
    variance_of_laplacian will exceed the default threshold of 60.0.
    """
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return img


def _unit_vec(dim: int = EMBEDDING_DIM, seed: int = 0) -> np.ndarray:
    """Return a deterministic unit-norm float32 vector of length ``dim``."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)



# Task 8.1 — EmbeddingGenerator unit tests


# ---------------------------------------------------------------------------
# Test 1: generate_embedding returns unit-norm float32 of shape (EMBEDDING_DIM,)


def test_generate_embedding_returns_unit_norm_float32():
    """
    When the mocked InsightFace model returns a non-zero embedding, the result
    must be float32, shape (EMBEDDING_DIM,), and have L2-norm within 1e-6 of 1.0.
    """
    raw_emb = _unit_vec() * 5.0  # non-unit length — generator must normalise it
    gen = make_mock_generator(mock_embedding=raw_emb)

    # Give the face detector a valid detection that passes the face-size gate.
    valid_detection = FaceDetection(x=10, y=10, width=80, height=80, confidence=0.99)
    gen._face_detector.detect.return_value = [valid_detection]

    image = _make_sharp_bgr()

    with patch("embedding_generator.variance_of_laplacian", return_value=200.0):
        result = gen.generate_embedding(image)

    assert result.shape == (EMBEDDING_DIM,), f"Expected ({EMBEDDING_DIM},), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-6, f"L2 norm should be 1.0, got {norm}"


# ---------------------------------------------------------------------------
# Test 2: generate_embedding raises ValueError on zero-size array


def test_generate_embedding_raises_on_zero_size():
    """A (0, 0, 3) array has size == 0; must raise ValueError."""
    gen = make_mock_generator()
    with pytest.raises(ValueError):
        gen.generate_embedding(np.zeros((0, 0, 3), dtype=np.uint8))



# Test 3: generate_embedding raises ValueError on wrong ndim (2-D array)


def test_generate_embedding_raises_on_wrong_ndim():
    """A 2-D array must trigger ValueError (needs 3-D H×W×C)."""
    gen = make_mock_generator()
    with pytest.raises(ValueError):
        gen.generate_embedding(np.zeros((100, 100), dtype=np.uint8))



# Test 4: generate_embedding raises ValueError on wrong channel count (1-channel)


def test_generate_embedding_raises_on_wrong_channels():
    """A (H, W, 1) array has the wrong channel count; must raise ValueError."""
    gen = make_mock_generator()
    with pytest.raises(ValueError):
        gen.generate_embedding(np.zeros((100, 100, 1), dtype=np.uint8))



# Test 5: generate_embeddings returns [] when all images are blurry


def test_generate_embeddings_returns_empty_list_when_all_blurry(tmp_path):
    """
    When every image fails the blur check (variance_of_laplacian < threshold),
    generate_embeddings must return an empty list.
    """
    # Write one dummy .jpg so the directory is non-empty
    jpg = tmp_path / "frame_001.jpg"
    # Create a minimal valid JPEG so cv2.imread doesn't return None
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(jpg), dummy)

    gen = make_mock_generator()

    with patch("embedding_generator.variance_of_laplacian", return_value=0.0):
        result = gen.generate_embeddings(str(tmp_path))

    assert result == [], f"Expected [], got {result}"



# Test 6: generate_embeddings skips unreadable files (cv2.imread returns None)


def test_generate_embeddings_skips_unreadable_file(tmp_path):
    """
    When cv2.imread returns None for one file and a valid image for another,
    the unreadable file must be skipped; the result list has exactly 1 entry.
    """
    import cv2

    # Write two placeholder files so the directory listing picks them up
    bad_file = tmp_path / "bad_001.jpg"
    good_file = tmp_path / "good_002.jpg"
    bad_file.write_bytes(b"not an image")

    # Good file needs real image data for the mock to work correctly
    good_img = _make_sharp_bgr(100, 100)
    cv2.imwrite(str(good_file), good_img)

    raw_emb = _unit_vec()
    gen = make_mock_generator(mock_embedding=raw_emb)

    # Face detector returns a valid, large detection for any call
    valid_detection = FaceDetection(x=10, y=10, width=80, height=80, confidence=0.99)
    gen._face_detector.detect.return_value = [valid_detection]

    call_count = 0
    orig_imread = cv2.imread

    def selective_imread(path, *args, **kwargs):
        if os.path.basename(path) == "bad_001.jpg":
            return None
        return orig_imread(path, *args, **kwargs)

    with patch("embedding_generator.cv2.imread", side_effect=selective_imread):
        with patch("embedding_generator.variance_of_laplacian", return_value=200.0):
            result = gen.generate_embeddings(str(tmp_path))

    assert len(result) == 1, f"Expected 1 embedding (bad file skipped), got {len(result)}"



# Test 7: _aggregate_embeddings returns unit-norm vector for ≥1 valid embeddings


def test_aggregate_embeddings_returns_unit_norm():
    """
    Aggregating 10 random unit-norm vectors must produce a result with
    L2-norm within 1e-6 of 1.0.
    """
    gen = make_mock_generator()
    embeddings = [_unit_vec(seed=i) for i in range(10)]

    result = gen._aggregate_embeddings(embeddings, "TestStudent", "999")

    assert result is not None, "_aggregate_embeddings should not return None for valid input"
    assert result.dtype == np.float32
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-6, f"L2 norm should be 1.0, got {norm}"



# Test 8: _aggregate_embeddings returns None when mean has zero norm


def test_aggregate_embeddings_returns_none_for_zero_norm():
    """
    Two equal-magnitude, opposite-sign unit vectors cancel out to a zero mean.
    _aggregate_embeddings must detect this and return None.
    """
    gen = make_mock_generator()
    v = _unit_vec(seed=7)
    neg_v = (-v).astype(np.float32)

    result = gen._aggregate_embeddings([v, neg_v], "TestStudent", "999")

    assert result is None, f"Expected None for zero-norm mean, got {result}"


# Task 8.2 — EmbeddingStore unit tests



# Helper to build a minimal StudentRecord with a unit-norm representative
# embedding.


def _make_record(roll: str = "001", name: str = "Alice") -> StudentRecord:
    rep = _unit_vec(seed=int(roll) if roll.isdigit() else 42)
    return StudentRecord(
        roll_number=roll,
        name=name,
        individual_embeddings=[rep],
        representative_embedding=rep,
    )


# ---------------------------------------------------------------------------
# Test 9: save_embeddings raises ValueError on empty list


def test_save_embeddings_raises_on_empty_list(tmp_path):
    """Passing an empty list to save_embeddings must raise ValueError."""
    filepath = str(tmp_path / "emb.pkl")
    with pytest.raises(ValueError):
        EmbeddingStore.save_embeddings([], filepath)


# ---------------------------------------------------------------------------
# Test 10: load_embeddings returns {} for a non-existent file


def test_load_embeddings_returns_empty_dict_for_missing_file(tmp_path):
    """If the pickle file doesn't exist, load_embeddings must return {}."""
    filepath = str(tmp_path / "nonexistent.pkl")
    result = EmbeddingStore.load_embeddings(filepath)
    assert result == {}, f"Expected {{}}, got {result}"



# Test 11: load_embeddings raises IOError for a corrupt file


def test_load_embeddings_raises_ioerror_for_corrupt_file(tmp_path):
    """A file containing garbage bytes must cause load_embeddings to raise IOError."""
    filepath = tmp_path / "corrupt.pkl"
    filepath.write_bytes(b"not a valid pickle")
    with pytest.raises(IOError):
        EmbeddingStore.load_embeddings(str(filepath))


# ---------------------------------------------------------------------------
# Test 12: add_student raises ValueError on duplicate roll_number


def test_add_student_raises_on_duplicate(tmp_path):
    """Adding a student whose roll_number already exists must raise ValueError."""
    filepath = str(tmp_path / "store.pkl")
    record = _make_record(roll="042")

    EmbeddingStore.add_student(record, filepath)

    with pytest.raises(ValueError):
        EmbeddingStore.add_student(record, filepath)



# Test 13: update_student raises KeyError on missing roll_number


def test_update_student_raises_on_missing(tmp_path):
    """Updating a roll_number not present in the store must raise KeyError."""
    filepath = str(tmp_path / "store.pkl")
    # Store is intentionally empty (file not created yet)
    record = _make_record(roll="999")

    with pytest.raises(KeyError):
        EmbeddingStore.update_student(record, filepath)


# Test 14: delete_student is a no-op (no exception) for missing roll_number


def test_delete_student_noop_for_missing(tmp_path):
    """
    Deleting a non-existent roll_number must not raise any exception and must
    leave the store file unchanged (or absent if it was never created).
    """
    filepath = str(tmp_path / "store.pkl")

    # Optionally seed the store with a different student
    other = _make_record(roll="001")
    EmbeddingStore.add_student(other, filepath)

    mtime_before = os.path.getmtime(filepath)

    # delete a non-existent roll — should be silent
    EmbeddingStore.delete_student("nonexistent_roll", filepath)

    # File must still exist and be unmodified (no rewrite for no-op)
    assert os.path.exists(filepath), "Store file should still exist after no-op delete"
    mtime_after = os.path.getmtime(filepath)
    assert mtime_before == mtime_after, "File should not be rewritten on no-op delete"



# Test 15: get_student returns None for a missing roll_number


def test_get_student_returns_none_for_missing(tmp_path):
    """get_student must return None when roll_number is not in the store."""
    filepath = str(tmp_path / "store.pkl")
    # File doesn't exist — should still return None, not raise
    result = EmbeddingStore.get_student("nonexistent", filepath)
    assert result is None, f"Expected None, got {result}"



# Test 16: backward compatibility — individual_embeddings defaults to []
#          when loading an old-format record that lacks the attribute.


def test_backward_compat_individual_embeddings_defaults_to_empty_list(tmp_path):
    """
    If a pickled StudentRecord lacks the individual_embeddings attribute
    (simulating an old store format), load_embeddings must default it to [].
    """
    filepath = str(tmp_path / "old_store.pkl")

    # Build a record and remove the individual_embeddings attr to simulate the
    # old format before that field was added.
    record = _make_record(roll="007", name="OldStudent")
    del record.individual_embeddings  # simulate old pickle without this field

    old_store = {"007": record}
    with open(filepath, "wb") as fh:
        fh.write(pickle.dumps(old_store))

    loaded = EmbeddingStore.load_embeddings(filepath)

    assert "007" in loaded, "Record for roll '007' should be present"
    loaded_record = loaded["007"]
    assert hasattr(loaded_record, "individual_embeddings"), \
        "individual_embeddings should have been patched in by load_embeddings"
    assert loaded_record.individual_embeddings == [], \
        f"Expected [], got {loaded_record.individual_embeddings}"
