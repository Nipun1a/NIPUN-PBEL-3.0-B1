"""
Unit tests for backend/services/recognition_service.py.

Tests:
  - process_frame: valid JPEG bytes call the recognizer and return results
  - process_frame: invalid bytes raise ValueError
  - process_frame: oversized / corrupt input raises ValueError
  - reload_embeddings: calls r.reload_embeddings() and returns len(r._store)
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import backend.services.recognition_service as recognition_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_jpeg() -> bytes:
    """Return a minimal valid 1×1 pixel JPEG as bytes."""
    import cv2
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".jpg", img)
    assert success, "cv2.imencode failed to produce a JPEG"
    return buf.tobytes()


def _make_mock_recognizer(store_size: int = 3):
    """Return a MagicMock that mimics the Recognizer interface."""
    mock = MagicMock()
    mock._store = {str(i): object() for i in range(store_size)}
    return mock


# ---------------------------------------------------------------------------
# process_frame tests
# ---------------------------------------------------------------------------

class TestProcessFrame:

    def test_valid_jpeg_calls_recognizer_and_returns_result(self):
        """Valid JPEG bytes should decode successfully and delegate to Recognizer."""
        mock_recognizer = _make_mock_recognizer()
        expected_result = ([], np.zeros((1, 1, 3), dtype=np.uint8))
        mock_recognizer.process_frame.return_value = expected_result

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            result = recognition_service.process_frame(_make_valid_jpeg())

        mock_recognizer.process_frame.assert_called_once()
        assert result is expected_result

    def test_invalid_bytes_raise_value_error(self):
        """Bytes that are not a valid image must raise ValueError."""
        mock_recognizer = _make_mock_recognizer()

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            with pytest.raises(ValueError, match="Could not decode"):
                recognition_service.process_frame(b"not a valid jpeg")

    def test_empty_bytes_raise_value_error(self):
        """Empty bytes must raise ValueError (not a valid image)."""
        mock_recognizer = _make_mock_recognizer()

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(b"")

    def test_oversized_corrupt_bytes_raise_value_error(self):
        """Large random bytes that don't form a valid image must raise ValueError."""
        mock_recognizer = _make_mock_recognizer()
        corrupt_bytes = b"\xff\xd8\xff" + b"\x00" * 10_000  # JPEG magic but corrupt body

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(corrupt_bytes)

    def test_process_frame_passes_numpy_array_to_recognizer(self):
        """Decoded frame passed to process_frame must be a numpy ndarray."""
        mock_recognizer = _make_mock_recognizer()
        mock_recognizer.process_frame.return_value = ([], None)
        captured = {}

        def capture_call(frame):
            captured["frame"] = frame
            return ([], None)

        mock_recognizer.process_frame.side_effect = capture_call

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            recognition_service.process_frame(_make_valid_jpeg())

        assert isinstance(captured["frame"], np.ndarray)

    def test_recognizer_not_initialised_raises_runtime_error(self):
        """process_frame must raise RuntimeError when no recognizer is set."""
        with patch.object(recognition_service, "_recognizer", None):
            with pytest.raises(RuntimeError, match="not initialised"):
                recognition_service.process_frame(_make_valid_jpeg())


# ---------------------------------------------------------------------------
# reload_embeddings tests
# ---------------------------------------------------------------------------

class TestReloadEmbeddings:

    def test_calls_reload_on_recognizer(self):
        """reload_embeddings() must call r.reload_embeddings() exactly once."""
        mock_recognizer = _make_mock_recognizer(store_size=2)

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            recognition_service.reload_embeddings()

        mock_recognizer.reload_embeddings.assert_called_once()

    def test_returns_store_length(self):
        """reload_embeddings() must return the number of entries in r._store."""
        mock_recognizer = _make_mock_recognizer(store_size=5)

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            count = recognition_service.reload_embeddings()

        assert count == 5

    def test_returns_zero_for_empty_store(self):
        """reload_embeddings() returns 0 when _store is empty."""
        mock_recognizer = _make_mock_recognizer(store_size=0)

        with patch.object(recognition_service, "_recognizer", mock_recognizer):
            count = recognition_service.reload_embeddings()

        assert count == 0

    def test_reload_not_called_when_recognizer_uninitialised(self):
        """reload_embeddings() raises RuntimeError before calling anything on recognizer."""
        with patch.object(recognition_service, "_recognizer", None):
            with pytest.raises(RuntimeError, match="not initialised"):
                recognition_service.reload_embeddings()
