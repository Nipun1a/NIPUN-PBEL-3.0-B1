"""
recognizer.py — Real-Time Face Recognition orchestration module.

Wires detection → crop → resize → embed → match → smooth → annotate for every
video frame and exposes a clean, framework-agnostic process_frame API.

This file is built incrementally across spec tasks:
  Task 3.1 — SmoothingBuffer (add, mark_stale, is_expired)
  Task 3.2 — SmoothingBuffer.vote()
  Task 4   — Recognizer.__init__ and reload_embeddings
  Task 5   — Recognizer.process_frame
  Task 7   — Recognizer.run_webcam
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional, List, Tuple

import numpy as np

# RecognitionResult is defined in face_matcher.py (created in task 1).
# Guard the import so this file can be loaded before face_matcher.py exists.
try:
    from face_matcher import RecognitionResult, FaceMatcher  # noqa: F401
except ImportError:  # pragma: no cover
    RecognitionResult = None  # type: ignore[assignment,misc]
    FaceMatcher = None  # type: ignore[assignment,misc]

try:
    from face_detector import FaceDetector, FaceDetection
except ImportError:  # pragma: no cover
    FaceDetector = None  # type: ignore[assignment,misc]
    FaceDetection = None  # type: ignore[assignment,misc]

try:
    from embedding_generator import EmbeddingGenerator, EmbeddingStore
except ImportError:  # pragma: no cover
    EmbeddingGenerator = None  # type: ignore[assignment,misc]
    EmbeddingStore = None  # type: ignore[assignment,misc]

try:
    from config import EMBEDDINGS_FILE, RECOGNITION_THRESHOLD, FACE_SIZE, BLUR_THRESHOLD, MIN_FACE_SIZE
except ImportError:  # pragma: no cover
    EMBEDDINGS_FILE = "embeddings.pkl"
    RECOGNITION_THRESHOLD = 0.6
    FACE_SIZE = (160, 160)
    BLUR_THRESHOLD = 50.0
    MIN_FACE_SIZE = 60

try:
    from quality import variance_of_laplacian
except ImportError:  # pragma: no cover
    variance_of_laplacian = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger("recognizer")


class SmoothingBuffer:
    """Per-face-track rolling window of recent RecognitionResult objects.

    Suppresses prediction flickering by collecting results over a configurable
    window and exposing a majority-vote result via vote() (added in task 3.2).

    Requirements: 9.1, 9.6
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialise the buffer.

        Args:
            window_size: Maximum number of RecognitionResult objects retained.
                         Older entries are evicted automatically (FIFO) once the
                         deque reaches capacity.  Requirement 9.1.
        """
        # collections.deque with maxlen enforces the bounded window (Req 9.1).
        self._buffer: deque = deque(maxlen=window_size)

        # Bounding-box centre of the tracked face, updated on every add() call.
        self.centre: Tuple[float, float] = (0.0, 0.0)

        # Number of consecutive frames since the last successful add() call.
        # Used for eviction — see is_expired() and Requirement 9.6.
        self._frames_since_update: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, result: "RecognitionResult") -> None:  # type: ignore[name-defined]
        """Append a new RecognitionResult to the rolling window.

        Also updates self.centre from result.bounding_box and resets the
        staleness counter to 0.

        Args:
            result: The RecognitionResult for this frame's detection.
        """
        self._buffer.append(result)
        self._frames_since_update = 0

        # Derive the bounding-box centre from (x, y, width, height).
        x, y, w, h = result.bounding_box
        self.centre = (x + w / 2.0, y + h / 2.0)

    def mark_stale(self) -> None:
        """Increment the staleness counter by one frame.

        Called by Recognizer._evict_stale_buffers() for every buffer that was
        not matched to any detection in the current frame (Requirement 9.6).
        """
        self._frames_since_update += 1

    def is_expired(self, ttl: int) -> bool:
        """Return True if this buffer has not been updated for more than ttl frames.

        A buffer is considered expired (and should be evicted) when
        _frames_since_update strictly exceeds ttl (Requirement 9.6).

        Args:
            ttl: Maximum allowed frames without an update before expiry.

        Returns:
            True if the buffer should be evicted, False otherwise.
        """
        return self._frames_since_update > ttl

    def vote(self):  # -> RecognitionResult
        """Return the majority-vote RecognitionResult from the current window.

        Majority-vote algorithm (Requirements 9.2, 9.3):
          1. Count occurrences of each (recognition_status, name) pair.
          2. Select the identity with the highest count.
          3. On a tie, select the identity with the highest *average*
             confidence_score among the tied identities.
          4. Return a RecognitionResult with the winning identity and the
             average confidence_score of that identity's entries in the buffer.

        Special cases:
          - Empty buffer: returns Unknown with confidence 0.0 and bbox (0,0,0,0).
          - Bounding box: taken from the most recent entry in the buffer.
        """
        # Empty buffer — Requirement 9.2
        if not self._buffer:
            return RecognitionResult(
                name="Unknown",
                roll_number="",
                confidence_score=0.0,
                recognition_status="Unknown",
                bounding_box=(0, 0, 0, 0),
            )

        # Step 1 — count occurrences and accumulate confidence scores per identity key.
        # Identity key: (recognition_status, name, roll_number) — name alone could collide
        # across statuses, so we group by the full triplet.
        from collections import defaultdict

        counts: dict = defaultdict(int)
        confidence_sums: dict = defaultdict(float)
        # Store a representative record per key (name, roll_number, status)
        identity_meta: dict = {}

        for result in self._buffer:
            key = (result.recognition_status, result.name, result.roll_number)
            counts[key] += 1
            confidence_sums[key] += result.confidence_score
            identity_meta[key] = result  # last seen; used for metadata fallback

        # Step 2 — find the maximum vote count
        max_count = max(counts.values())

        # Collect all keys tied at max_count
        tied_keys = [k for k, v in counts.items() if v == max_count]

        # Step 3 — break ties by highest average confidence_score
        def avg_confidence(key):
            return confidence_sums[key] / counts[key]

        winning_key = max(tied_keys, key=avg_confidence)

        # Step 4 — build the result
        w_status, w_name, w_roll = winning_key
        avg_conf = avg_confidence(winning_key)

        # Bounding box from the most recent entry in the buffer
        latest_bbox = self._buffer[-1].bounding_box

        return RecognitionResult(
            name=w_name,
            roll_number=w_roll,
            confidence_score=avg_conf,
            recognition_status=w_status,
            bounding_box=latest_bbox,
        )


# ---------------------------------------------------------------------------
# Recognizer class — Requirements 1.1, 1.2, 1.3, 2.3, 4.2, 10.2, 10.4
# ---------------------------------------------------------------------------

class Recognizer:
    """
    Orchestrates the full per-frame recognition pipeline:
    detection → crop → resize → embed → match → smooth → annotate.

    Exposes a clean, framework-agnostic API consumable by FastAPI/Flask
    route handlers or a standalone webcam loop.

    Args:
        embeddings_file:       Path to the pickle embeddings store.
        recognition_threshold: Cosine similarity threshold for Known/Unknown.
        smoothing_window:      Number of frames in the per-track vote window.
        tracking_distance:     Max pixel distance to reuse an existing buffer.
        buffer_ttl:            Frames of inactivity before a buffer is evicted.
        apply_blur_filter:     Reject blurry faces before embedding generation.
        face_margin:           Fractional margin to expand face crops.
        detector:              Optional pre-constructed FaceDetector instance.
        generator:             Optional pre-constructed EmbeddingGenerator instance.
    """

    def __init__(
        self,
        embeddings_file: str = EMBEDDINGS_FILE,
        recognition_threshold: float = RECOGNITION_THRESHOLD,
        smoothing_window: int = 5,
        tracking_distance: float = 80.0,
        buffer_ttl: int = 30,
        apply_blur_filter: bool = False,
        face_margin: float = 0.15,
        detector: Optional["FaceDetector"] = None,
        generator: Optional["EmbeddingGenerator"] = None,
    ) -> None:
        # Store configuration parameters as instance attributes (Req 10.2)
        self._embeddings_file = embeddings_file
        self._recognition_threshold = recognition_threshold
        self._smoothing_window = smoothing_window
        self._tracking_distance = tracking_distance
        self._buffer_ttl = buffer_ttl
        self._apply_blur_filter = apply_blur_filter
        self._face_margin = face_margin

        # Lazy component injection — Req 2.3, 4.2, 10.4:
        # Accept externally provided instances (useful for testing and sharing)
        # or construct one here if none is given.
        if detector is not None:
            self._detector = detector
        else:
            self._detector = FaceDetector()

        if generator is not None:
            self._generator = generator
        else:
            self._generator = EmbeddingGenerator()

        # Load the embeddings store exactly once at init time (Req 1.1)
        self._store = EmbeddingStore.load_embeddings(embeddings_file)

        # Warn if the store is empty so the operator knows all faces will be
        # classified Unknown until the store is populated (Req 1.2, 12.4)
        if not self._store:
            logger.warning(
                "Embeddings store at '%s' is empty — all faces will be "
                "classified as Unknown until the store is populated.",
                embeddings_file,
            )

        # Per-face tracking state
        self._buffers: List[SmoothingBuffer] = []

        # Cosine-similarity matcher
        self._matcher = FaceMatcher(recognition_threshold)

    # ------------------------------------------------------------------
    # reload_embeddings — Requirements 1.4, 12.5
    # ------------------------------------------------------------------

    def reload_embeddings(self) -> None:
        """Reload the in-memory embeddings store from disk.

        Requirements: 1.4, 12.5
        """
        self._store = EmbeddingStore.load_embeddings(self._embeddings_file)
        logger.info(
            "reload_embeddings: loaded %d student record(s) from '%s'.",
            len(self._store),
            self._embeddings_file,
        )

    # ------------------------------------------------------------------
    # process_frame — Requirements 2.1, 2.2, 2.4, 3.1–3.5, 4.1, 4.3,
    #                 7.1–7.3, 8.1–8.6, 9.4–9.6, 10.1, 12.3, 12.6
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> "Tuple[List[RecognitionResult], np.ndarray]":
        """Run the full per-frame recognition pipeline.

        Args:
            frame: BGR numpy array representing the current video frame.

        Returns:
            A tuple of (results, annotated_frame) where results is a list of
            RecognitionResult objects (one per detection) and annotated_frame
            is a copy of the input frame with bounding boxes and labels drawn.

        Requirements: 2.1, 2.2, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.3,
                      7.1, 7.2, 7.3, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6,
                      9.4, 9.5, 9.6, 10.1, 12.3, 12.6
        """
        try:
            # Step 1 — detect faces (Req 2.1, 2.4)
            try:
                detections = self._detector.detect(frame)
            except Exception as exc:
                logger.error(
                    "process_frame: FaceDetector.detect raised an exception: %s", exc
                )
                return [], frame

            # Req 2.2 — empty detection list → return unchanged
            if not detections:
                return [], frame

            results: "List[RecognitionResult]" = []
            matched_buffers: set = set()

            for det in detections:
                # Step 1 — size check (Req 3.2)
                if det.width < MIN_FACE_SIZE or det.height < MIN_FACE_SIZE:
                    logger.debug(
                        "process_frame: face too small (%dx%d) at bbox %s — Unknown",
                        det.width, det.height, det.box,
                    )
                    results.append(
                        RecognitionResult(
                            name="Unknown",
                            roll_number="",
                            confidence_score=0.0,
                            recognition_status="Unknown",
                            bounding_box=det.box,
                        )
                    )
                    continue

                # Step 2 — crop (Req 3.1)
                face_crop = self._detector.crop_face(frame, det, margin=self._face_margin)

                # Step 3 — optional blur check (Req 3.5)
                if self._apply_blur_filter and variance_of_laplacian(face_crop) < BLUR_THRESHOLD:
                    logger.debug(
                        "process_frame: face at bbox %s rejected by blur filter — Unknown",
                        det.box,
                    )
                    results.append(
                        RecognitionResult(
                            name="Unknown",
                            roll_number="",
                            confidence_score=0.0,
                            recognition_status="Unknown",
                            bounding_box=det.box,
                        )
                    )
                    continue

                # Step 4 — resize to FACE_SIZE (Req 3.3)
                face_resized = cv2.resize(face_crop, FACE_SIZE)

                # Step 5 — generate embedding (Req 4.1, 4.3, 12.3)
                try:
                    embedding = self._generator.generate_embedding(
                        face_resized, min_quality_score=0.0
                    )
                except ValueError as exc:
                    logger.warning(
                        "process_frame: embedding generation failed for bbox %s: %s",
                        det.box, exc,
                    )
                    results.append(
                        RecognitionResult(
                            name="Unknown",
                            roll_number="",
                            confidence_score=0.0,
                            recognition_status="Unknown",
                            bounding_box=det.box,
                        )
                    )
                    continue

                # Step 6 — match against stored embeddings
                raw_result = self._matcher.match(embedding, self._store, det.box)

                # Step 7 — smooth via per-track buffer (Req 9.4, 9.5)
                buffer = self._find_or_create_buffer(det)
                matched_buffers.add(id(buffer))
                buffer.add(raw_result)
                smoothed_result = buffer.vote()

                # Step 8 — collect result
                results.append(smoothed_result)

            # Annotate a copy of the frame (Req 8.4, 8.5)
            annotated = self._annotate_frame(frame.copy(), detections, results)

            # Evict stale buffers (Req 9.6)
            self._evict_stale_buffers(matched_buffers)

            return results, annotated

        except Exception as exc:
            # Req 12.6 — log unhandled exceptions at ERROR level and re-raise
            logger.error(
                "process_frame: unhandled exception: %s", exc, exc_info=True
            )
            raise

    # ------------------------------------------------------------------
    # _find_or_create_buffer — Requirements 9.4, 9.5
    # ------------------------------------------------------------------

    def _find_or_create_buffer(self, det: "FaceDetection") -> SmoothingBuffer:
        """Find the nearest existing buffer or create a new one for ``det``.

        Computes the bounding-box centre (cx, cy) of ``det`` and searches
        ``self._buffers`` for the nearest buffer whose centre is within
        ``self._tracking_distance`` Euclidean pixels.  If found, that buffer's
        centre is updated to (cx, cy) and its staleness counter is reset to 0.
        If none is close enough, a new SmoothingBuffer is created and appended.

        Requirements: 9.4, 9.5
        """
        cx = det.x + det.width / 2.0
        cy = det.y + det.height / 2.0

        best_buf = None
        best_dist = float("inf")

        for buf in self._buffers:
            bx, by = buf.centre
            dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_buf = buf

        if best_buf is not None and best_dist <= self._tracking_distance:
            best_buf.centre = (cx, cy)
            best_buf._frames_since_update = 0
            return best_buf

        # No buffer within range — create a new one (Req 9.5)
        new_buf = SmoothingBuffer(self._smoothing_window)
        new_buf.centre = (cx, cy)
        self._buffers.append(new_buf)
        return new_buf

    # ------------------------------------------------------------------
    # _evict_stale_buffers — Requirement 9.6
    # ------------------------------------------------------------------

    def _evict_stale_buffers(self, matched_buffers: set) -> None:
        """Mark unmatched buffers as stale and remove expired ones.

        For every buffer NOT in ``matched_buffers``, ``mark_stale()`` is
        called to increment its staleness counter.  Buffers whose staleness
        counter exceeds ``self._buffer_ttl`` are then removed.

        Args:
            matched_buffers: Set of ``id(buffer)`` values for buffers that
                             were matched to a detection this frame.

        Requirements: 9.6
        """
        for buf in self._buffers:
            if id(buf) not in matched_buffers:
                buf.mark_stale()

        self._buffers = [
            buf for buf in self._buffers
            if not buf.is_expired(self._buffer_ttl)
        ]

    # ------------------------------------------------------------------
    # _annotate_frame — Requirements 8.1–8.6
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # run_webcam — Requirements 11.1, 11.2, 11.3, 11.4
    # ------------------------------------------------------------------

    def run_webcam(self, camera_index: int = 0, show_fps: bool = True) -> None:
        """
        Open a webcam, run process_frame per frame, display annotated result.
        Requirements: 11.1, 11.2, 11.3, 11.4
        """
        import time
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")  # Req 11.4

        try:
            prev_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, annotated = self.process_frame(frame)

                if show_fps:  # Req 11.2
                    now = time.time()
                    fps = int(1.0 / (now - prev_time + 1e-9))
                    prev_time = now
                    cv2.putText(annotated, str(fps), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Face Recognition", annotated)  # Req 11.1

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Req 11.3
                    break
        finally:
            cap.release()           # Req 11.3
            cv2.destroyAllWindows() # Req 11.3

    # ------------------------------------------------------------------
    # _annotate_frame — Requirements 8.1–8.6
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        annotated: np.ndarray,
        detections: "List[FaceDetection]",
        results: "List[RecognitionResult]",
    ) -> np.ndarray:
        """Draw bounding boxes and identity labels on ``annotated``.

        Draws on the frame that was passed in (callers should pass
        ``frame.copy()`` to preserve the original — Req 8.4).

        Color coding (Req 8.1, 8.2):
          - Known face  → green BGR (0, 255, 0)
          - Unknown face → red BGR (0, 0, 255)

        Label format (Req 8.3): ``"{name} ({confidence_score:.2f})"``

        Label placement (Req 8.6): drawn above the bounding box; if the
        label would extend above the top of the frame it is drawn below the
        top edge of the box instead.

        Args:
            annotated: BGR numpy array to draw on (already a copy).
            detections: List of FaceDetection objects from the detector.
            results:    List of RecognitionResult objects, one per detection,
                        in the same positional order.

        Returns:
            The annotated frame (same object that was passed in).

        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        box_thickness = 2

        for det, result in zip(detections, results):
            # Choose box colour (Req 8.1, 8.2)
            if result.recognition_status == "Known":
                color = (0, 255, 0)   # green — Known
            else:
                color = (0, 0, 255)   # red — Unknown

            x, y, w, h = det.box

            # Draw bounding box (Req 8.1, 8.2)
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                color,
                box_thickness,
            )

            # Build label text (Req 8.3)
            label = f"{result.name} ({result.confidence_score:.2f})"

            # Measure label dimensions to handle boundary placement (Req 8.6)
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            label_height = label_h + baseline

            # Determine vertical position: above box unless that would clip (Req 8.6)
            if y - label_height < 0:
                # Draw below the top edge of the box
                label_y = y + label_h
            else:
                # Draw above the bounding box
                label_y = y - baseline

            cv2.putText(
                annotated,
                label,
                (x, label_y),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated


if __name__ == "__main__":
    Recognizer().run_webcam()
