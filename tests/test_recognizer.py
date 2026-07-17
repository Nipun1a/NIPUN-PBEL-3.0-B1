"""
tests/test_recognizer.py

Unit tests for the recognizer.py pipeline components:
  - SmoothingBuffer: bounded window, staleness tracking, expiry, vote()
  - Recognizer.process_frame: empty detections, N detections, exceptions,
    embedding ValueError, original-frame immutability, annotation colour
  - Recognizer.__init__: custom detector/generator injection
  - Recognizer.reload_embeddings: INFO log emitted
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_detector import FaceDetection
from face_matcher import RecognitionResult
from recognizer import SmoothingBuffer, Recognizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(x=10, y=10, w=80, h=80, conf=0.9):
    """Construct a FaceDetection with given pixel coordinates."""
    return FaceDetection(x=x, y=y, width=w, height=h, confidence=conf)


def make_result(
    name="Alice",
    roll="101",
    score=0.85,
    status="Known",
    bbox=(10, 10, 80, 80),
) -> RecognitionResult:
    return RecognitionResult(
        name=name,
        roll_number=roll,
        confidence_score=score,
        recognition_status=status,
        bounding_box=bbox,
    )


def unknown_result(bbox=(10, 10, 80, 80)) -> RecognitionResult:
    return RecognitionResult(
        name="Unknown",
        roll_number="",
        confidence_score=0.0,
        recognition_status="Unknown",
        bounding_box=bbox,
    )


# ---------------------------------------------------------------------------
# SmoothingBuffer — window-size bounding
# ---------------------------------------------------------------------------

class TestSmoothingBufferBounded:
    def test_buffer_bounded_by_window(self):
        """Deque must never grow past window_size."""
        buf = SmoothingBuffer(window_size=3)
        for i in range(10):
            buf.add(make_result(bbox=(i, i, 80, 80)))
        assert len(buf._buffer) <= 3


# ---------------------------------------------------------------------------
# SmoothingBuffer — staleness tracking
# ---------------------------------------------------------------------------

class TestSmoothingBufferStaleness:
    def test_add_resets_staleness(self):
        """After add(), _frames_since_update must be 0."""
        buf = SmoothingBuffer()
        buf.mark_stale()
        buf.mark_stale()
        buf.add(make_result())
        assert buf._frames_since_update == 0

    def test_mark_stale_increments(self):
        """Each mark_stale() call increments _frames_since_update by 1."""
        buf = SmoothingBuffer()
        assert buf._frames_since_update == 0
        buf.mark_stale()
        assert buf._frames_since_update == 1
        buf.mark_stale()
        assert buf._frames_since_update == 2


# ---------------------------------------------------------------------------
# SmoothingBuffer — expiry
# ---------------------------------------------------------------------------

class TestSmoothingBufferExpiry:
    def test_is_expired_false(self):
        """Buffer with 0 stale frames and ttl=5 should not be expired."""
        buf = SmoothingBuffer()
        # _frames_since_update starts at 0 — not expired
        assert buf.is_expired(ttl=5) is False

    def test_is_expired_true(self):
        """Buffer that has been marked stale ttl+1 times should be expired."""
        buf = SmoothingBuffer()
        ttl = 5
        for _ in range(ttl + 1):
            buf.mark_stale()
        assert buf.is_expired(ttl=ttl) is True


# ---------------------------------------------------------------------------
# SmoothingBuffer.vote() — edge cases and majority
# ---------------------------------------------------------------------------

class TestSmoothingBufferVote:
    def test_vote_empty(self):
        """vote() on an empty buffer returns Unknown with confidence 0.0."""
        buf = SmoothingBuffer()
        result = buf.vote()
        assert result.name == "Unknown"
        assert result.recognition_status == "Unknown"
        assert result.confidence_score == pytest.approx(0.0)

    def test_vote_majority(self):
        """3x Alice, 1x Unknown → winner is Alice."""
        buf = SmoothingBuffer(window_size=10)
        for _ in range(3):
            buf.add(make_result(name="Alice", roll="101", status="Known"))
        buf.add(unknown_result())
        result = buf.vote()
        assert result.name == "Alice"
        assert result.recognition_status == "Known"

    def test_vote_tiebreak(self):
        """Tie: Alice score=0.9, Bob score=0.5 → Alice wins (higher avg confidence)."""
        buf = SmoothingBuffer(window_size=10)
        buf.add(make_result(name="Alice", roll="101", score=0.9, status="Known"))
        buf.add(make_result(name="Bob", roll="102", score=0.5, status="Known"))
        result = buf.vote()
        assert result.name == "Alice"

    def test_vote_updates_bbox_from_last(self):
        """vote() bounding_box must come from the last added result."""
        buf = SmoothingBuffer(window_size=5)
        buf.add(make_result(bbox=(0, 0, 50, 50)))
        buf.add(make_result(bbox=(100, 200, 60, 70)))  # last
        result = buf.vote()
        assert result.bounding_box == (100, 200, 60, 70)


# ---------------------------------------------------------------------------
# Recognizer helpers — build a minimally wired instance without real models
# ---------------------------------------------------------------------------

def _make_recognizer(
    detections=None,
    embedding=None,
    store=None,
):
    """
    Return a Recognizer with mocked FaceDetector and EmbeddingGenerator so
    no real model loading happens.

    Parameters
    ----------
    detections : list | None
        Return value for detector.detect(). Defaults to [].
    embedding : np.ndarray | None
        Return value for generator.generate_embedding(). Defaults to a
        zero-norm-safe unit vector.
    store : dict | None
        Embeddings store. Defaults to {}.
    """
    mock_detector = MagicMock()
    mock_detector.detect.return_value = detections if detections is not None else []
    mock_detector.crop_face.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

    mock_generator = MagicMock()
    if embedding is None:
        v = np.ones(512, dtype=np.float32)
        embedding = (v / np.linalg.norm(v)).astype(np.float32)
    mock_generator.generate_embedding.return_value = embedding

    with patch("recognizer.EmbeddingStore") as mock_es:
        mock_es.load_embeddings.return_value = store if store is not None else {}
        rec = Recognizer(
            embeddings_file="dummy.pkl",
            detector=mock_detector,
            generator=mock_generator,
        )

    # Replace the store directly so matcher has access
    rec._store = store if store is not None else {}
    return rec, mock_detector, mock_generator


def _blank_frame(h=240, w=320):
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Recognizer.process_frame — empty detections
# ---------------------------------------------------------------------------

class TestProcessFrameEmptyDetections:
    def test_process_frame_empty_detections(self):
        """detector returns [] → process_frame returns ([], original_frame)."""
        rec, _, _ = _make_recognizer(detections=[])
        frame = _blank_frame()
        results, out_frame = rec.process_frame(frame)
        assert results == []
        # Returned frame should be the same object (unchanged)
        assert out_frame is frame


# ---------------------------------------------------------------------------
# Recognizer.process_frame — N detections → N results
# ---------------------------------------------------------------------------

class TestProcessFrameNDetections:
    def test_process_frame_n_detections_n_results(self):
        """N FaceDetection objects → len(results) == N."""
        for n in [1, 2, 3]:
            dets = [make_det(x=i * 100, y=10, w=80, h=80) for i in range(n)]
            rec, _, _ = _make_recognizer(detections=dets)
            frame = _blank_frame(h=480, w=640)
            results, _ = rec.process_frame(frame)
            assert len(results) == n, f"Expected {n} results, got {len(results)}"


# ---------------------------------------------------------------------------
# Recognizer.process_frame — exception handling
# ---------------------------------------------------------------------------

class TestProcessFrameExceptions:
    def test_process_frame_detector_exception(self):
        """detector.detect raises Exception → returns ([], frame), no re-raise."""
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("camera error")
        mock_generator = MagicMock()

        with patch("recognizer.EmbeddingStore") as mock_es:
            mock_es.load_embeddings.return_value = {}
            rec = Recognizer(
                embeddings_file="dummy.pkl",
                detector=mock_detector,
                generator=mock_generator,
            )
        rec._store = {}

        frame = _blank_frame()
        # Must not raise
        results, out_frame = rec.process_frame(frame)
        assert results == []
        assert out_frame is frame

    def test_process_frame_embedding_valueerror(self):
        """EmbeddingGenerator raises ValueError → that face is Unknown, no re-raise."""
        det = make_det()
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [det]
        mock_detector.crop_face.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

        mock_generator = MagicMock()
        mock_generator.generate_embedding.side_effect = ValueError("bad image")

        with patch("recognizer.EmbeddingStore") as mock_es:
            mock_es.load_embeddings.return_value = {}
            rec = Recognizer(
                embeddings_file="dummy.pkl",
                detector=mock_detector,
                generator=mock_generator,
            )
        rec._store = {}

        frame = _blank_frame()
        results, _ = rec.process_frame(frame)
        assert len(results) == 1
        assert results[0].recognition_status == "Unknown"
        assert results[0].name == "Unknown"


# ---------------------------------------------------------------------------
# Recognizer.process_frame — original frame immutability
# ---------------------------------------------------------------------------

class TestProcessFrameImmutability:
    def test_process_frame_original_not_mutated(self):
        """process_frame must not alter the original frame's byte content."""
        det = make_det()
        rec, _, _ = _make_recognizer(detections=[det])
        frame = _blank_frame()
        snapshot = frame.copy()
        rec.process_frame(frame)
        assert np.array_equal(frame, snapshot), "Original frame was mutated by process_frame"


# ---------------------------------------------------------------------------
# Recognizer.process_frame — Known result → green bounding box pixels
# ---------------------------------------------------------------------------

class TestProcessFrameAnnotation:
    def test_process_frame_known_result_green_box(self):
        """Mock returning a Known result → annotated frame has green pixels at bbox border."""
        bbox = (10, 10, 80, 80)
        det = make_det(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [det]
        mock_detector.crop_face.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

        v = np.ones(512, dtype=np.float32)
        unit_emb = (v / np.linalg.norm(v)).astype(np.float32)
        mock_generator = MagicMock()
        mock_generator.generate_embedding.return_value = unit_emb

        # Patch the matcher on the recognizer to return a Known result directly
        with patch("recognizer.EmbeddingStore") as mock_es:
            mock_es.load_embeddings.return_value = {}
            rec = Recognizer(
                embeddings_file="dummy.pkl",
                detector=mock_detector,
                generator=mock_generator,
            )
        rec._store = {}

        # Patch matcher.match to return Known
        known_result = RecognitionResult(
            name="Alice",
            roll_number="101",
            confidence_score=0.9,
            recognition_status="Known",
            bounding_box=bbox,
        )
        rec._matcher = MagicMock()
        rec._matcher.match.return_value = known_result

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        results, annotated = rec.process_frame(frame)

        # Check the top border of the bounding box for green pixels
        x, y, w, h = bbox
        # cv2.rectangle draws the top edge along row y, cols x to x+w
        # Check a pixel roughly in the middle of the top edge
        mid_x = x + w // 2
        pixel = annotated[y, mid_x]  # BGR
        assert pixel[1] > 200, (
            f"Expected green channel > 200 at top border of bbox, got {pixel}"
        )
        assert pixel[0] < 10, f"Expected blue channel ~0, got {pixel}"
        assert pixel[2] < 10, f"Expected red channel ~0, got {pixel}"


# ---------------------------------------------------------------------------
# Recognizer.__init__ — custom detector/generator injection
# ---------------------------------------------------------------------------

class TestRecognizerInit:
    def test_recognizer_accepts_custom_detector_generator(self):
        """Injected mock objects must be stored as _detector and _generator."""
        mock_detector = MagicMock()
        mock_generator = MagicMock()

        with patch("recognizer.EmbeddingStore") as mock_es:
            mock_es.load_embeddings.return_value = {}
            rec = Recognizer(
                embeddings_file="dummy.pkl",
                detector=mock_detector,
                generator=mock_generator,
            )

        assert rec._detector is mock_detector
        assert rec._generator is mock_generator


# ---------------------------------------------------------------------------
# Recognizer.reload_embeddings — INFO log
# ---------------------------------------------------------------------------

class TestReloadEmbeddings:
    def test_reload_embeddings_logs_info(self, caplog):
        """reload_embeddings() must emit at least one INFO log."""
        with patch("recognizer.EmbeddingStore") as mock_es:
            mock_es.load_embeddings.return_value = {}
            rec = Recognizer(
                embeddings_file="dummy.pkl",
                detector=MagicMock(),
                generator=MagicMock(),
            )
            with caplog.at_level(logging.INFO, logger="recognizer"):
                rec.reload_embeddings()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1, (
            "Expected at least one INFO log from reload_embeddings(), got none"
        )
