"""
dataset_collector.py

Reusable face-dataset collection service.

This module has NO dependency on any particular UI or web framework.
It exposes DatasetCollector.collect(), a single method that:
    - opens the camera
    - detects faces frame-by-frame (MediaPipe)
    - validates quality (exactly one face, sharp enough, large enough)
    - saves cropped, resized faces to disk with sequential names
    - reports progress via an optional callback

A FastAPI/Flask endpoint can call `DatasetCollector().collect(...)`
directly (e.g. from a background task/thread) and stream progress to
the frontend via the `progress_callback` instead of relying on the
OpenCV preview window, which only makes sense for a local/kiosk app.
"""

import os
import time
import logging
from typing import Callable, Optional, Tuple

import cv2

try:
    from .config import (
        DATASET_ROOT,
        IMAGES_PER_STUDENT,
        FACE_SIZE,
        MIN_DETECTION_CONFIDENCE,
        BLUR_THRESHOLD,
        MIN_FACE_SIZE,
        CAPTURE_DELAY,
        POSE_INSTRUCTIONS,
    )
    from .face_detector import FaceDetector
    from .quality import is_blurry, is_face_too_small
    from .models import StudentDetails, CaptureResult
except ImportError:  # Allow running this module directly as a script
    from config import (
        DATASET_ROOT,
        IMAGES_PER_STUDENT,
        FACE_SIZE,
        MIN_DETECTION_CONFIDENCE,
        BLUR_THRESHOLD,
        MIN_FACE_SIZE,
        CAPTURE_DELAY,
        POSE_INSTRUCTIONS,
    )
    from face_detector import FaceDetector
    from quality import is_blurry, is_face_too_small
    from models import StudentDetails, CaptureResult

logger = logging.getLogger(__name__)

# Signature: (student, images_saved, target_images, status_message) -> None
ProgressCallback = Callable[[StudentDetails, int, int, str], None]


class DatasetCollector:
    """
    Reusable, framework-agnostic face dataset collection service.

    Usage (local script):
        collector = DatasetCollector()
        result = collector.collect(StudentDetails(name="Yug", roll_number="102"))

    Usage (future backend API):
        collector = DatasetCollector()
        result = collector.collect(
            student=student_from_request,
            show_preview=False,
            progress_callback=push_progress_over_websocket,
        )
    """

    def __init__(
        self,
        dataset_root: str = DATASET_ROOT,
        images_per_student: int = IMAGES_PER_STUDENT,
        face_size: Tuple[int, int] = FACE_SIZE,
        blur_threshold: float = BLUR_THRESHOLD,
        min_face_size: int = MIN_FACE_SIZE,
        capture_delay: float = CAPTURE_DELAY,
    ):
        self.dataset_root = dataset_root
        self.images_per_student = images_per_student
        self.face_size = face_size
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size
        self.capture_delay = capture_delay

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def collect(
        self,
        student: StudentDetails,
        camera_index: int = 0,
        show_preview: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> CaptureResult:
        """
        Run a full capture session for one student.

        Args:
            student: Student details (name, roll number, department, ...).
                     Never hardcoded - always passed in by the caller
                     (CLI, backend API, etc).
            camera_index: OpenCV camera device index.
            show_preview: If True, opens an OpenCV window with a live
                          preview, bounding box, progress and pose text.
                          Set to False for headless/server execution.
            progress_callback: Optional callback invoked after every
                          processed frame with
                          (student, saved_count, target, status_message).
                          Use this to push progress to a frontend via
                          websockets/SSE in the future.

        Returns:
            CaptureResult summarizing how many images were saved.
        """
        folder_path = self._prepare_student_folder(student)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            message = "Could not open camera."
            logger.error(message)
            return CaptureResult(student, 0, self.images_per_student, folder_path, False, message)

        saved_count = 0
        pose_index = 0
        images_per_pose = max(1, self.images_per_student // len(POSE_INSTRUCTIONS))

        try:
            with FaceDetector(min_detection_confidence=MIN_DETECTION_CONFIDENCE) as detector:
                logger.info("Starting capture for %s (%s)", student.name, student.roll_number)

                while saved_count < self.images_per_student:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame from camera.")
                        continue

                    frame = cv2.flip(frame, 1)  # mirror for a natural-feeling preview
                    detections = detector.detect(frame)

                    status_message, face_crop = self._evaluate_frame(detector, frame, detections)

                    if face_crop is not None:
                        self._save_face(folder_path, student, saved_count + 1, face_crop)
                        saved_count += 1
                        pose_index = min(saved_count // images_per_pose, len(POSE_INSTRUCTIONS) - 1)
                        time.sleep(self.capture_delay)

                    if progress_callback:
                        progress_callback(student, saved_count, self.images_per_student, status_message)

                    if show_preview:
                        self._render_preview(
                            frame, detections, student, saved_count,
                            POSE_INSTRUCTIONS[pose_index], status_message,
                        )
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("Capture cancelled by user.")
                            break
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        completed = saved_count >= self.images_per_student
        final_message = "Capture complete." if completed else "Capture stopped early."
        logger.info("%s %s: %d/%d saved", final_message, student.name, saved_count, self.images_per_student)

        return CaptureResult(student, saved_count, self.images_per_student, folder_path, completed, final_message)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _prepare_student_folder(self, student: StudentDetails) -> str:
        """Create (if needed) and return the folder where this student's images are stored."""
        folder_path = os.path.join(self.dataset_root, student.folder_name())
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _evaluate_frame(self, detector: FaceDetector, frame, detections):
        """
        Apply all validation rules to a frame's detections.

        Returns:
            (status_message, cropped_and_resized_face_or_None)
            face is None whenever the frame should NOT be saved.
        """
        if len(detections) == 0:
            return "No face detected", None

        if len(detections) > 1:
            return "Multiple faces detected - only one person allowed", None

        detection = detections[0]

        if is_face_too_small(detection.width, detection.height, self.min_face_size):
            return "Face too far from camera", None

        face_crop = detector.crop_face(frame, detection)

        if face_crop.size == 0:
            return "Invalid face crop", None

        if is_blurry(face_crop, self.blur_threshold):
            return "Image too blurry - hold still", None

        resized_face = cv2.resize(face_crop, self.face_size)
        return "Good frame captured", resized_face

    def _save_face(self, folder_path: str, student: StudentDetails, index: int, face_image) -> None:
        """Save a validated face crop with a sequential filename, e.g. Yug_045.jpg."""
        filename = f"{student.name}_{index:03d}.jpg"
        filepath = os.path.join(folder_path, filename)
        cv2.imwrite(filepath, face_image)

    def _render_preview(self, frame, detections, student, saved_count, pose_instruction, status_message) -> None:
        """Draw bounding box(es), progress text, and pose instruction onto the preview frame."""
        display_frame = frame.copy()

        for det in detections:
            x, y, w, h = det.box
            # Green when exactly one face (the "good" case), red otherwise
            color = (0, 255, 0) if len(detections) == 1 else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

        progress_text = f"{student.name} - {saved_count}/{self.images_per_student}"
        cv2.putText(display_frame, progress_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Pose: {pose_instruction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, status_message, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Dataset Collection", display_frame)