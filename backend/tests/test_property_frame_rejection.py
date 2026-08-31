"""
Property test for frame rejection before ML (Task 8.3).

Tests that process_frame raises ValueError for invalid / oversized payloads
BEFORE the Recognizer is ever called.

Validates: Requirements 13.1, 13.2, 13.3
"""
from __future__ import annotations

import sys
import os
import random

import pytest
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.services import recognition_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_bytes(size: int) -> bytes:
    """Return *size* random bytes that are not a valid JPEG."""
    rng = random.Random(42)
    return bytes(rng.getrandbits(8) for _ in range(size))


# ---------------------------------------------------------------------------
# Tests: invalid bytes rejected before Recognizer is called
# ---------------------------------------------------------------------------

class TestFrameRejectionBeforeML:
    """
    Property: process_frame must raise ValueError for any non-decodable input
    without ever calling Recognizer.process_frame.
    """

    def test_empty_bytes_raises_value_error(self):
        """Empty payload → ValueError, recognizer never called."""
        mock_recognizer = MagicMock()

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError, match="Could not decode"):
                recognition_service.process_frame(b"")

        mock_recognizer.process_frame.assert_not_called()

    def test_random_noise_bytes_raises_value_error(self):
        """Random non-JPEG bytes → ValueError, recognizer never called."""
        mock_recognizer = MagicMock()
        bad_bytes = _make_random_bytes(1024)

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError, match="Could not decode"):
                recognition_service.process_frame(bad_bytes)

        mock_recognizer.process_frame.assert_not_called()

    def test_small_random_bytes_raises_value_error(self):
        """Small (16-byte) random payload → ValueError, recognizer never called."""
        mock_recognizer = MagicMock()
        bad_bytes = _make_random_bytes(16)

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(bad_bytes)

        mock_recognizer.process_frame.assert_not_called()

    @pytest.mark.parametrize(
        "payload",
        [
            b"\x00" * 100,           # null bytes — not a JPEG
            b"\xff" * 200,           # all 0xFF — not a valid JPEG body
            b"NOT_AN_IMAGE_AT_ALL",  # ASCII text
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,  # PNG magic header, not JPEG
            b"RIFF" + b"\x00" * 100,               # RIFF/WAV magic header
        ],
        ids=[
            "null_bytes",
            "all_ff_bytes",
            "ascii_text",
            "png_header",
            "riff_header",
        ],
    )
    def test_various_invalid_payloads_rejected(self, payload):
        """
        Property: any non-JPEG bytes must be rejected with ValueError and
        the recognizer's process_frame method must not be called.
        """
        mock_recognizer = MagicMock()

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(payload)

        mock_recognizer.process_frame.assert_not_called()

    def test_oversized_random_payload_raises_value_error(self):
        """
        Oversized payload (10 MB of random bytes, not a valid image) →
        ValueError, recognizer never called.
        """
        mock_recognizer = MagicMock()
        oversized = _make_random_bytes(10 * 1024 * 1024)  # 10 MB

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(oversized)

        mock_recognizer.process_frame.assert_not_called()

    def test_oversized_null_payload_raises_value_error(self):
        """
        Oversized payload of null bytes → ValueError, recognizer never called.
        Validates oversized detection independent of random content.
        """
        mock_recognizer = MagicMock()
        oversized = b"\x00" * (5 * 1024 * 1024)  # 5 MB of zeros

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(oversized)

        mock_recognizer.process_frame.assert_not_called()

    def test_truncated_jpeg_header_raises_value_error(self):
        """
        A JPEG magic header (0xFF 0xD8 0xFF) without a valid body must still
        fail cv2.imdecode and raise ValueError before reaching the Recognizer.
        """
        mock_recognizer = MagicMock()
        # JPEG magic bytes followed by garbage
        truncated_jpeg = b"\xff\xd8\xff" + b"\x00" * 50

        with patch.object(recognition_service, "get_recognizer", return_value=mock_recognizer):
            with pytest.raises(ValueError):
                recognition_service.process_frame(truncated_jpeg)

        mock_recognizer.process_frame.assert_not_called()
