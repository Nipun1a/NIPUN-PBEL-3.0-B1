"""
embedding_generator.py

Face embedding generation module for the Attendance Monitoring System.

Model: InsightFace ArcFace (buffalo_l variant)
Embedding dimensionality: EMBEDDING_DIM = 512

This module loads a pre-trained InsightFace ArcFace model once at startup,
validates image quality (blur, face presence, face size), generates 512-d
L2-normalised float32 embeddings per image, aggregates them into a per-student
representative embedding, and persists the results to an embeddings store.

It exposes a clean, framework-agnostic Python API callable directly from
FastAPI or Flask route handlers, returning structured EmbeddingOperationResult
objects that serialise to JSON without any adapter code.

Requirements: 1.4, 8.1, 9.1, 10.1, 10.2
"""

import os
import pickle
import logging
import dataclasses
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Two-step config import — mirrors the pattern from dataset_collector.py
# ---------------------------------------------------------------------------
try:
    from .config import (
        DATASET_ROOT,
        FACE_SIZE,
        BLUR_THRESHOLD,
        MIN_FACE_SIZE,
        EMBEDDINGS_FILE,
        MIN_QUALITY_SCORE,
        RECOGNITION_THRESHOLD,
    )
    from .models import StudentDetails
    from .face_detector import FaceDetector
    from .quality import variance_of_laplacian
except ImportError:  # Allow running this module directly as a script
    from config import (
        DATASET_ROOT,
        FACE_SIZE,
        BLUR_THRESHOLD,
        MIN_FACE_SIZE,
        EMBEDDINGS_FILE,
        MIN_QUALITY_SCORE,
        RECOGNITION_THRESHOLD,
    )
    from models import StudentDetails
    from face_detector import FaceDetector
    from quality import variance_of_laplacian

# ---------------------------------------------------------------------------
# Module-level logger — Requirement 9.1
# ---------------------------------------------------------------------------
logger = logging.getLogger("embedding_generator")

# ---------------------------------------------------------------------------
# Module-level constant — Requirement 1.4
# InsightFace ArcFace buffalo_l produces 512-dimensional embeddings.
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 512

# ---------------------------------------------------------------------------
# Custom exception — Requirement 8.4
# ---------------------------------------------------------------------------

class EmbeddingError(Exception):
    """
    Raised by public API functions when an unrecoverable error occurs.

    Attributes:
        message (str): Human-readable description of the failure.
        __cause__:     The originating exception, set via
                       ``raise EmbeddingError(...) from cause``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Data classes — Requirements 11.1, 13.11
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StudentRecord:
    """
    Persisted unit for one student's identity.

    Both ``individual_embeddings`` (one float32 array per accepted image, in
    ascending lexicographic filename order) and ``representative_embedding``
    (mean L2-normalised vector across all accepted images) are stored so that
    the recognition layer can choose its matching strategy.

    Invariants:
        - ``roll_number`` is never ``None`` or empty string.
        - Every entry in ``individual_embeddings`` has dtype=float32 and
          shape=(EMBEDDING_DIM,) with L2-norm ≈ 1.0.
        - ``representative_embedding`` (when not None) has dtype=float32,
          shape=(EMBEDDING_DIM,), and L2-norm within 1e-6 of 1.0.
    """

    roll_number: str
    name: str
    individual_embeddings: List[np.ndarray] = dataclasses.field(
        default_factory=list
    )
    representative_embedding: Optional[np.ndarray] = None


@dataclasses.dataclass
class EmbeddingOperationResult:
    """
    Structured result returned by all public CRUD API functions.

    Designed for direct JSON serialisation by FastAPI/Flask route handlers —
    the caller never needs to catch raw exceptions from the public API layer.

    Fields:
        success (bool):               Whether the operation succeeded.
        message (str):                Human-readable status or error message.
        roll_number (str):            The student's roll number involved.
        student_name (Optional[str]): The student's name when available.
    """

    success: bool
    message: str
    roll_number: str
    student_name: Optional[str] = None


# ---------------------------------------------------------------------------
# EmbeddingGenerator class — Requirements 1.1, 1.2, 1.3
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    Loads a pre-trained face recognition model once and exposes per-image
    and per-student-folder embedding generation with integrated quality gating.

    Model is loaded in __init__; all subsequent calls reuse the same instance.
    Raises RuntimeError on init if the required package is absent.
    """

    def __init__(
        self,
        face_size: tuple = FACE_SIZE,
        min_quality_score: float = MIN_QUALITY_SCORE,
        min_face_size: int = MIN_FACE_SIZE,
    ) -> None:
        self._face_size = face_size
        self._min_quality_score = min_quality_score
        self._min_face_size = min_face_size

        # Try InsightFace first (preferred — ArcFace, 512-d)
        try:
            import insightface  # noqa: F401
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._model = app
            self._use_insightface = True
            logger.info("EmbeddingGenerator: loaded InsightFace ArcFace (buffalo_l)")
        except ImportError:
            # Fall back to facenet-pytorch
            try:
                import facenet_pytorch  # noqa: F401
                from facenet_pytorch import InceptionResnetV1

                self._model = InceptionResnetV1(pretrained="vggface2").eval()
                self._use_insightface = False
                logger.info("EmbeddingGenerator: loaded FaceNet (facenet-pytorch, vggface2)")
            except ImportError:
                raise RuntimeError(
                    "Face recognition model not found. Install one of:\n"
                    "  pip install insightface onnxruntime\n"
                    "  pip install facenet-pytorch"
                )

        # Instantiate the face detector (used by generate_embeddings for quality gating)
        self._face_detector = FaceDetector()

    def generate_embedding(self, image, min_quality_score=None):
        """
        Generate an L2-normalised float32 embedding from a single BGR face image.

        Parameters
        ----------
        image : np.ndarray
            BGR image array of shape (H, W, 3).
        min_quality_score : float or None
            Override the instance-level quality threshold for this call only.
            If None, ``self._min_quality_score`` is used.

        Returns
        -------
        np.ndarray
            float32 array of shape (EMBEDDING_DIM,) with L2-norm ≈ 1.0.

        Raises
        ------
        ValueError
            If the image array shape is invalid, quality is too low, no face is
            detected (InsightFace backend), or a zero-norm embedding is produced.

        Requirements: 2.1, 2.2, 2.3, 2.4
        """
        # ------------------------------------------------------------------
        # 1. Input validation — Requirement 2.3
        # ------------------------------------------------------------------
        if image.ndim != 3:
            raise ValueError(
                f"image.ndim must be 3, got {image.ndim}"
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"image.shape[2] must be 3 (BGR channels), got {image.shape[2]}"
            )
        if image.size == 0:
            raise ValueError("image.size must be > 0, got 0 (empty array)")

        # ------------------------------------------------------------------
        # 2. Quality gate — Requirement 12.3, 12.8
        # ------------------------------------------------------------------
        threshold = min_quality_score if min_quality_score is not None else self._min_quality_score
        score = variance_of_laplacian(image)
        if score < threshold:
            raise ValueError(f"Image quality too low: {score:.2f}")

        # ------------------------------------------------------------------
        # 3. Pre-processing — Requirement 2.2
        #    BGR → RGB, resize to FACE_SIZE, normalise pixels to [-1, 1]
        #    The input array is never mutated.
        # ------------------------------------------------------------------
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self._face_size)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0

        # ------------------------------------------------------------------
        # 4. Model forward pass — handle both backends
        # ------------------------------------------------------------------
        if self._use_insightface:
            # CollectedImages/ contains pre-cropped face images. Bypass
            # InsightFace's detector (app.get) and call the recognition model
            # directly via get_feat, which expects a list of uint8 BGR images.
            # cv2.dnn.blobFromImages (called inside get_feat) handles
            # normalisation internally — do NOT pre-normalise here.
            rec_model = self._model.models.get("recognition")
            if rec_model is not None:
                # Resize to the model's required input size (112×112) as uint8
                input_size = rec_model.input_size  # e.g. (112, 112)
                face_resized = cv2.resize(image, input_size,
                                          interpolation=cv2.INTER_LINEAR)
                # get_feat expects a list of BGR uint8 images
                embedding = rec_model.get_feat([face_resized]).flatten().astype(np.float32)
            else:
                # Fallback: use app.get() on full-size photos
                faces = self._model.get(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if len(faces) == 0:
                    raise ValueError("No face detected in image")
                embedding = faces[0].embedding.astype(np.float32)
        else:
            import torch
            tensor = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                raw = self._model(tensor)
            embedding = raw[0].numpy().astype(np.float32)

        # ------------------------------------------------------------------
        # 5. L2 normalisation — Requirement 2.4
        # ------------------------------------------------------------------
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            raise ValueError("Zero-norm embedding produced")

        return (embedding / norm).astype(np.float32)

    def generate_embeddings(self, student_folder_path, min_quality_score=None):
        """
        Generate embeddings for all .jpg images in a student folder.

        Reads all .jpg files (case-insensitive) in lexicographic order,
        applies quality gating (blur check, face detection, face size check),
        and returns a list of L2-normalised float32 embedding vectors for
        all accepted images.

        Args:
            student_folder_path (str): Path to the directory containing the
                student's .jpg face images.
            min_quality_score (float | None): Override for the blur threshold.
                If None, falls back to ``self._min_quality_score``.

        Returns:
            list[np.ndarray]: Per-image float32 embeddings of shape
                ``(EMBEDDING_DIM,)``, in ascending lexicographic filename
                order.  Returns an empty list when all images are rejected.

        Raises:
            FileNotFoundError: If ``student_folder_path`` does not exist or
                is not a directory.

        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 12.3, 12.4, 12.5, 12.6,
                      12.7, 12.8
        """
        # Requirement 3.2 — validate the directory path
        if not os.path.isdir(student_folder_path):
            raise FileNotFoundError(
                f"Directory not found: {student_folder_path}"
            )

        # Resolve the effective quality threshold for this call (Req 12.8)
        effective_quality_score = (
            min_quality_score if min_quality_score is not None
            else self._min_quality_score
        )

        # Requirement 3.1 — collect .jpg files, sort lexicographically
        jpg_files = sorted(
            [
                f
                for f in os.listdir(student_folder_path)
                if f.lower().endswith(".jpg")
            ]
        )

        # Counters for per-reason rejection tracking
        accepted: List[np.ndarray] = []
        decode_errors = 0
        blur_count = 0
        small_face_count = 0
        no_face_count = 0

        for filename in jpg_files:
            filepath = os.path.join(student_folder_path, filename)

            # Step (b) — decode
            image = cv2.imread(filepath)
            if image is None:
                logger.warning("%s: decode failure", filename)
                decode_errors += 1
                continue

            # Step (c) — blur / quality gate (Req 12.3, 12.4)
            quality_score = variance_of_laplacian(image)
            if quality_score < effective_quality_score:
                logger.warning(
                    "%s: blur %.2f < threshold",
                    filename,
                    quality_score,
                )
                blur_count += 1
                continue

            # Step (d) — face detection (Req 12.6)
            detections = self._face_detector.detect(image)
            if not detections:
                logger.warning("%s: no face detected", filename)
                no_face_count += 1
                continue

            # Step (e) — face size gate (Req 12.5)
            best_det = detections[0]
            if (
                best_det.width < self._min_face_size
                or best_det.height < self._min_face_size
            ):
                logger.warning(
                    "%s: face too small (%dx%d)",
                    filename,
                    best_det.width,
                    best_det.height,
                )
                small_face_count += 1
                continue

            # Step (f) — generate embedding; pass min_quality_score=0.0 to
            # skip the redundant quality re-check inside generate_embedding
            # since we already validated quality above.
            embedding = self.generate_embedding(image, min_quality_score=0.0)
            accepted.append(embedding)

        # Step 5 — summary log (Req 3.4, 12.7)
        total_rejected = decode_errors + blur_count + small_face_count + no_face_count
        logger.info(
            "accepted=%d, rejected=%d, blur: %d, face too small: %d, no face detected: %d",
            len(accepted),
            total_rejected,
            blur_count,
            small_face_count,
            no_face_count,
        )

        # Step 6 — warn and return empty list if nothing was accepted (Req 3.5)
        if len(accepted) == 0:
            logger.warning(
                "All images rejected or failed to decode in %s",
                student_folder_path,
            )
            return []

        return accepted

    def _aggregate_embeddings(self, embeddings, student_name, roll_number):
        """
        Aggregate a list of per-image embeddings into a single representative embedding.

        Steps:
          1. Warn if fewer than 10 images were accepted.
          2. Compute the element-wise mean across all embeddings.
          3. L2-normalise the mean vector.
          4. Return the unit-norm float32 vector, or None if the mean is a zero vector.

        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        # Requirement 5.4 — warn when fewer than 10 images are available
        if len(embeddings) < 10:
            logger.warning(
                f"Only {len(embeddings)} accepted images for {student_name} ({roll_number})"
            )

        # Requirement 5.1 — element-wise mean
        mean_vec = np.mean(embeddings, axis=0)

        # Requirement 5.2 / 5.3 — L2 normalise; guard against zero vector
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-6:
            logger.error(
                f"Zero-norm mean vector for {student_name} ({roll_number}) — skipping"
            )
            return None

        return (mean_vec / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# EmbeddingStore class — Requirements 6.1–6.8, 7.1–7.6, 11.3, 11.4, 11.6,
#                        13.1–13.7
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """
    Manages persistence of the {roll_number → StudentRecord} mapping to/from
    a pickle file using atomic temp-file-then-rename writes.

    All methods are stateless between calls — each method loads from disk,
    modifies in memory, and saves back. This avoids stale-cache bugs and
    keeps the API simple for backend route handlers.
    """

    # ------------------------------------------------------------------
    # Internal helper — Requirements 6.3, 6.8, 13.7
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write(store: dict, filepath: str) -> None:
        """
        Serialize ``store`` to a temp file and atomically rename it to
        ``filepath``.  If anything fails, the tmp file is deleted and an
        ``IOError`` is raised, leaving the original file intact.

        Parameters
        ----------
        store : dict
            The ``{roll_number: StudentRecord}`` mapping to persist.
        filepath : str
            Destination path for the pickle file.

        Raises
        ------
        IOError
            If the write or rename step fails.
        """
        tmp_path = filepath + ".tmp"
        try:
            with open(tmp_path, "wb") as fh:
                fh.write(pickle.dumps(store))
            os.replace(tmp_path, filepath)
        except Exception as exc:
            # Clean up the partial tmp file if it exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass  # Best-effort cleanup; raise the original error below
            raise IOError(
                f"Failed to atomically write embeddings to {filepath}"
            ) from exc

    # ------------------------------------------------------------------
    # save_embeddings — Requirements 6.1, 6.2, 6.3, 6.8, 11.3, 11.4
    # ------------------------------------------------------------------

    @staticmethod
    def save_embeddings(records: list, filepath: str) -> None:
        """
        Persist a list of ``StudentRecord`` objects to ``filepath``.

        Parameters
        ----------
        records : list[StudentRecord]
            Non-empty list of student records to save.
        filepath : str
            Destination path for the pickle store.

        Raises
        ------
        ValueError
            If ``records`` is empty.
        IOError
            If the file cannot be written atomically.

        Requirements: 6.1, 6.2, 6.3, 6.4, 6.8, 11.3, 11.4
        """
        if not records:
            raise ValueError("Cannot save empty records list")

        # Build the roll_number → StudentRecord mapping
        store = {r.roll_number: r for r in records}

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(parent_dir, exist_ok=True)

        EmbeddingStore._atomic_write(store, filepath)

    # ------------------------------------------------------------------
    # load_embeddings — Requirements 6.5, 6.6, 6.7, 11.6
    # ------------------------------------------------------------------

    @staticmethod
    def load_embeddings(filepath: str) -> dict:
        """
        Load the ``{roll_number → StudentRecord}`` mapping from ``filepath``.

        Returns an empty dict if the file does not exist.  For each loaded
        record, ``individual_embeddings`` is defaulted to ``[]`` if the
        attribute is missing (backward-compatibility with older store format).

        Parameters
        ----------
        filepath : str
            Path to the pickle store.

        Returns
        -------
        dict[str, StudentRecord]
            Mapping of roll_number to StudentRecord.  Empty if file absent.

        Raises
        ------
        IOError
            If the file exists but cannot be unpickled or read.

        Requirements: 6.5, 6.6, 6.7, 11.6
        """
        if not os.path.exists(filepath):
            return {}

        try:
            with open(filepath, "rb") as fh:
                store = pickle.load(fh)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            raise IOError(
                f"Failed to load embeddings from {filepath}"
            ) from exc

        # Backward-compatibility: default individual_embeddings to [] if missing
        for roll_number, record in store.items():
            if not hasattr(record, "individual_embeddings"):
                logger.warning(
                    "Record for %s missing 'individual_embeddings'; defaulting to []",
                    roll_number,
                )
                record.individual_embeddings = []

        return store

    # ------------------------------------------------------------------
    # add_student — Requirements 7.1, 13.1
    # ------------------------------------------------------------------

    @staticmethod
    def add_student(student_record: "StudentRecord", filepath: str) -> None:
        """
        Insert a new student record into the store.

        Parameters
        ----------
        student_record : StudentRecord
            The student to add.  ``roll_number`` must be non-empty.
        filepath : str
            Path to the pickle store.

        Raises
        ------
        ValueError
            If ``student_record.roll_number`` is empty or if the roll_number
            already exists in the store.

        Requirements: 7.1, 13.1
        """
        if not student_record.roll_number:
            raise ValueError("student_record.roll_number must not be empty")

        store = EmbeddingStore.load_embeddings(filepath)

        if student_record.roll_number in store:
            raise ValueError(
                f"Student with roll_number '{student_record.roll_number}' already exists"
            )

        store[student_record.roll_number] = student_record
        EmbeddingStore._atomic_write(store, filepath)

    # ------------------------------------------------------------------
    # update_student — Requirements 7.2, 13.2
    # ------------------------------------------------------------------

    @staticmethod
    def update_student(student_record: "StudentRecord", filepath: str) -> None:
        """
        Overwrite an existing student record in the store.

        Parameters
        ----------
        student_record : StudentRecord
            The updated student data.
        filepath : str
            Path to the pickle store.

        Raises
        ------
        KeyError
            If ``student_record.roll_number`` is not found in the store.

        Requirements: 7.2, 13.2
        """
        store = EmbeddingStore.load_embeddings(filepath)

        if student_record.roll_number not in store:
            raise KeyError(
                f"Student with roll_number '{student_record.roll_number}' not found"
            )

        store[student_record.roll_number] = student_record
        EmbeddingStore._atomic_write(store, filepath)

    # ------------------------------------------------------------------
    # delete_student — Requirements 7.3, 13.3
    # ------------------------------------------------------------------

    @staticmethod
    def delete_student(roll_number: str, filepath: str) -> None:
        """
        Remove a student record from the store.

        If the roll_number is not found, logs a WARNING and returns without
        modifying the file (no-op).

        Parameters
        ----------
        roll_number : str
            The roll number of the student to delete.
        filepath : str
            Path to the pickle store.

        Requirements: 7.3, 13.3
        """
        store = EmbeddingStore.load_embeddings(filepath)

        if roll_number not in store:
            logger.warning(
                "delete_student: roll_number '%s' not found in store — no-op",
                roll_number,
            )
            return

        del store[roll_number]
        EmbeddingStore._atomic_write(store, filepath)

    # ------------------------------------------------------------------
    # get_student — Requirements 7.4, 13.4
    # ------------------------------------------------------------------

    @staticmethod
    def get_student(roll_number: str, filepath: str):
        """
        Retrieve a single student record by roll number.

        Parameters
        ----------
        roll_number : str
            The roll number to look up.
        filepath : str
            Path to the pickle store.

        Returns
        -------
        StudentRecord or None
            The matching record, or ``None`` if not found.

        Requirements: 7.4, 13.4
        """
        store = EmbeddingStore.load_embeddings(filepath)
        return store.get(roll_number, None)

    # ------------------------------------------------------------------
    # list_students — Requirements 7.5, 13.5
    # ------------------------------------------------------------------

    @staticmethod
    def list_students(filepath: str) -> list:
        """
        Return the list of all roll numbers currently in the store.

        Returns an empty list if the file does not exist.

        Parameters
        ----------
        filepath : str
            Path to the pickle store.

        Returns
        -------
        list[str]
            Sorted list of roll_number strings.

        Requirements: 7.5, 13.5
        """
        store = EmbeddingStore.load_embeddings(filepath)
        return list(store.keys())


# ---------------------------------------------------------------------------
# Private helper — used by all public API functions that need to find a student
# folder by roll number
# ---------------------------------------------------------------------------

def _find_student_folder(dataset_root: str, roll_number: str) -> Optional[str]:
    """
    Scan ``dataset_root`` for a direct sub-folder whose name ends with
    ``_{roll_number}``.

    Parameters
    ----------
    dataset_root : str
        Root directory that contains per-student sub-folders.
    roll_number : str
        The roll number to search for (matched against the suffix
        ``_{roll_number}`` of each sub-folder name).

    Returns
    -------
    str or None
        Absolute path to the matching folder, or ``None`` if not found.
    """
    for entry in os.scandir(dataset_root):
        if entry.is_dir() and entry.name.endswith(f"_{roll_number}"):
            return entry.path
    return None


# ---------------------------------------------------------------------------
# Task 7.1 — generate_and_save_all
# Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.2, 9.4, 9.5, 9.6
# ---------------------------------------------------------------------------

def generate_and_save_all(
    dataset_root: str = DATASET_ROOT,
    output_filepath: str = EMBEDDINGS_FILE,
) -> int:
    """
    Generate embeddings for every student folder under ``dataset_root`` and
    persist the full set to ``output_filepath`` via :class:`EmbeddingStore`.

    Folder discovery:
        Only direct sub-folders whose names match the ``{Name}_{RollNumber}``
        pattern (split on the **last** underscore; both parts must be
        non-empty) are processed.  All other entries are skipped with a DEBUG
        log message.

    Parameters
    ----------
    dataset_root : str
        Root directory containing per-student sub-folders.
        Defaults to :data:`DATASET_ROOT`.
    output_filepath : str
        Destination path for the pickle store.
        Defaults to :data:`EMBEDDINGS_FILE`.

    Returns
    -------
    int
        Number of student records successfully saved.

    Raises
    ------
    FileNotFoundError
        If ``dataset_root`` does not exist.
    EmbeddingError
        Wraps any other fatal exception encountered during processing.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.2, 9.4, 9.5, 9.6
    """
    # Requirement 4.1 — validate dataset_root before traversal
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"dataset_root does not exist: {dataset_root}"
        )

    try:
        # Create one EmbeddingGenerator instance and reuse it for all students
        # (model weights are loaded only once — Requirement 1.2)
        generator = EmbeddingGenerator()

        records: List[StudentRecord] = []
        total_attempted = 0
        total_skipped = 0
        total_accepted_images = 0

        # Requirement 4.2 — traverse direct sub-folders
        for entry in sorted(os.scandir(dataset_root), key=lambda e: e.name):
            if not entry.is_dir():
                logger.debug("Skipping non-directory entry: %s", entry.name)
                continue

            # Requirement 4.3 — validate folder name pattern (split on LAST underscore)
            last_sep = entry.name.rfind("_")
            if last_sep <= 0 or last_sep == len(entry.name) - 1:
                # No underscore, underscore at position 0, or trailing underscore
                logger.debug(
                    "Skipping folder (does not match Name_RollNumber pattern): %s",
                    entry.name,
                )
                continue

            student_name = entry.name[:last_sep]
            roll_number = entry.name[last_sep + 1:]

            if not student_name or not roll_number:
                logger.debug(
                    "Skipping folder (empty name or roll_number after split): %s",
                    entry.name,
                )
                continue

            total_attempted += 1
            folder_path = entry.path

            # Requirement 4.4 — generate embeddings for this student
            embeddings = generator.generate_embeddings(folder_path)

            if not embeddings:
                logger.error(
                    "No accepted embeddings for %s (%s) — skipping",
                    student_name,
                    roll_number,
                )
                total_skipped += 1
                continue

            # Requirement 4.5 — aggregate into a representative embedding
            representative = generator._aggregate_embeddings(
                embeddings, student_name, roll_number
            )

            if representative is None:
                logger.error(
                    "Zero-norm aggregate for %s (%s) — skipping",
                    student_name,
                    roll_number,
                )
                total_skipped += 1
                continue

            records.append(
                StudentRecord(
                    roll_number=roll_number,
                    name=student_name,
                    individual_embeddings=embeddings,
                    representative_embedding=representative,
                )
            )
            total_accepted_images += len(embeddings)

        # Requirement 4.6 — persist all collected records
        if records:
            EmbeddingStore.save_embeddings(records, output_filepath)

        successful = len(records)

        # Requirements 9.4, 9.5, 9.6 — summary log
        logger.info(
            "generate_and_save_all summary: attempted=%d, successful=%d, "
            "skipped=%d, total_accepted_images=%d",
            total_attempted,
            successful,
            total_skipped,
            total_accepted_images,
        )

        return successful

    except FileNotFoundError:
        raise  # Re-raise without wrapping (already a clear error)
    except Exception as exc:
        raise EmbeddingError(
            f"Fatal error in generate_and_save_all: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Task 7.2 — generate_and_append_student
# Requirements: 8.3
# ---------------------------------------------------------------------------

def generate_and_append_student(
    student_details: "StudentDetails",
    dataset_root: str = DATASET_ROOT,
    output_filepath: str = EMBEDDINGS_FILE,
) -> StudentRecord:
    """
    Generate embeddings for a single student and upsert the record into the
    embedding store (add if new, overwrite if already present).

    Parameters
    ----------
    student_details : StudentDetails
        Name and roll number of the student to process.
    dataset_root : str
        Root directory containing per-student sub-folders.
        Defaults to :data:`DATASET_ROOT`.
    output_filepath : str
        Path to the pickle store.
        Defaults to :data:`EMBEDDINGS_FILE`.

    Returns
    -------
    StudentRecord
        The persisted student record (including both individual and
        representative embeddings).

    Raises
    ------
    EmbeddingError
        If the student folder cannot be found, all images are rejected,
        the aggregate is a zero vector, or any other fatal error occurs.

    Requirements: 8.3
    """
    try:
        # Locate the student's folder
        folder_path = _find_student_folder(dataset_root, student_details.roll_number)
        if folder_path is None:
            raise EmbeddingError(
                f"Student folder not found for roll_number '{student_details.roll_number}' "
                f"in '{dataset_root}'"
            )

        generator = EmbeddingGenerator()

        embeddings = generator.generate_embeddings(folder_path)
        if not embeddings:
            raise EmbeddingError(
                f"No accepted embeddings for {student_details.name} "
                f"({student_details.roll_number})"
            )

        representative = generator._aggregate_embeddings(
            embeddings, student_details.name, student_details.roll_number
        )
        if representative is None:
            raise EmbeddingError(
                f"Zero-norm aggregate embedding for {student_details.name} "
                f"({student_details.roll_number})"
            )

        record = StudentRecord(
            roll_number=student_details.roll_number,
            name=student_details.name,
            individual_embeddings=embeddings,
            representative_embedding=representative,
        )

        # Upsert: add if new, update if existing
        existing = EmbeddingStore.get_student(student_details.roll_number, output_filepath)
        if existing is not None:
            EmbeddingStore.update_student(record, output_filepath)
        else:
            EmbeddingStore.add_student(record, output_filepath)

        return record

    except EmbeddingError:
        raise  # Already the correct type
    except Exception as exc:
        raise EmbeddingError(
            f"Fatal error in generate_and_append_student for "
            f"'{student_details.roll_number}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Task 7.3 — add_new_student
# Requirements: 13.8, 13.12, 13.13
# ---------------------------------------------------------------------------

def add_new_student(
    student_details: "StudentDetails",
    dataset_root: str = DATASET_ROOT,
    output_filepath: str = EMBEDDINGS_FILE,
) -> EmbeddingOperationResult:
    """
    Generate embeddings for a student and add them to the store as a new
    record.  Raises :class:`ValueError` (wrapped as failure) if the student
    already exists.

    Parameters
    ----------
    student_details : StudentDetails
        Name and roll number of the student.
    dataset_root : str
        Root directory containing per-student sub-folders.
        Defaults to :data:`DATASET_ROOT`.
    output_filepath : str
        Path to the pickle store.
        Defaults to :data:`EMBEDDINGS_FILE`.

    Returns
    -------
    EmbeddingOperationResult
        ``success=True`` on success (message includes accepted image count),
        ``success=False`` on any failure (message contains error details).

    Requirements: 13.8, 13.12, 13.13
    """
    try:
        # Locate the student folder; treat missing folder as a failure result
        folder_path = _find_student_folder(dataset_root, student_details.roll_number)
        if folder_path is None:
            msg = (
                f"Student folder not found for roll_number "
                f"'{student_details.roll_number}' in '{dataset_root}'"
            )
            logger.error(
                "add_new_student [%s]: %s (%s)",
                student_details.roll_number,
                msg,
                "FileNotFoundError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=student_details.roll_number,
                student_name=student_details.name,
            )

        generator = EmbeddingGenerator()

        embeddings = generator.generate_embeddings(folder_path)
        if not embeddings:
            msg = (
                f"All images rejected for {student_details.name} "
                f"({student_details.roll_number})"
            )
            logger.error(
                "add_new_student [%s]: %s (%s)",
                student_details.roll_number,
                msg,
                "ValueError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=student_details.roll_number,
                student_name=student_details.name,
            )

        representative = generator._aggregate_embeddings(
            embeddings, student_details.name, student_details.roll_number
        )
        if representative is None:
            msg = (
                f"Zero-norm aggregate embedding for {student_details.name} "
                f"({student_details.roll_number})"
            )
            logger.error(
                "add_new_student [%s]: %s (%s)",
                student_details.roll_number,
                msg,
                "ValueError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=student_details.roll_number,
                student_name=student_details.name,
            )

        record = StudentRecord(
            roll_number=student_details.roll_number,
            name=student_details.name,
            individual_embeddings=embeddings,
            representative_embedding=representative,
        )

        EmbeddingStore.add_student(record, output_filepath)

        return EmbeddingOperationResult(
            success=True,
            message=(
                f"Successfully added {student_details.name} "
                f"({student_details.roll_number}) with "
                f"{len(embeddings)} accepted image(s)"
            ),
            roll_number=student_details.roll_number,
            student_name=student_details.name,
        )

    except Exception as exc:
        logger.error(
            "add_new_student [%s]: %s (%s)",
            student_details.roll_number,
            exc,
            type(exc).__name__,
        )
        return EmbeddingOperationResult(
            success=False,
            message=str(exc),
            roll_number=student_details.roll_number,
            student_name=student_details.name,
        )


# ---------------------------------------------------------------------------
# Task 7.4 — update_existing_student
# Requirements: 13.9, 13.12, 13.13
# ---------------------------------------------------------------------------

def update_existing_student(
    roll_number: str,
    dataset_root: str = DATASET_ROOT,
    output_filepath: str = EMBEDDINGS_FILE,
) -> EmbeddingOperationResult:
    """
    Re-generate embeddings for an existing student and overwrite their record
    in the store.

    Parameters
    ----------
    roll_number : str
        The roll number of the student to update.
    dataset_root : str
        Root directory containing per-student sub-folders.
        Defaults to :data:`DATASET_ROOT`.
    output_filepath : str
        Path to the pickle store.
        Defaults to :data:`EMBEDDINGS_FILE`.

    Returns
    -------
    EmbeddingOperationResult
        ``success=True`` on success, ``success=False`` on any failure.

    Requirements: 13.9, 13.12, 13.13
    """
    try:
        # Locate the student folder
        folder_path = _find_student_folder(dataset_root, roll_number)
        if folder_path is None:
            msg = (
                f"Student folder not found for roll_number '{roll_number}' "
                f"in '{dataset_root}'"
            )
            logger.error(
                "update_existing_student [%s]: %s (%s)",
                roll_number,
                msg,
                "FileNotFoundError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=roll_number,
            )

        # Derive the student name from the folder name (part before last underscore)
        folder_name = os.path.basename(folder_path)
        last_sep = folder_name.rfind("_")
        student_name = folder_name[:last_sep] if last_sep > 0 else folder_name

        generator = EmbeddingGenerator()

        embeddings = generator.generate_embeddings(folder_path)
        if not embeddings:
            msg = f"All images rejected for roll_number '{roll_number}'"
            logger.error(
                "update_existing_student [%s]: %s (%s)",
                roll_number,
                msg,
                "ValueError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=roll_number,
                student_name=student_name,
            )

        representative = generator._aggregate_embeddings(
            embeddings, student_name, roll_number
        )
        if representative is None:
            msg = f"Zero-norm aggregate embedding for roll_number '{roll_number}'"
            logger.error(
                "update_existing_student [%s]: %s (%s)",
                roll_number,
                msg,
                "ValueError",
            )
            return EmbeddingOperationResult(
                success=False,
                message=msg,
                roll_number=roll_number,
                student_name=student_name,
            )

        record = StudentRecord(
            roll_number=roll_number,
            name=student_name,
            individual_embeddings=embeddings,
            representative_embedding=representative,
        )

        EmbeddingStore.update_student(record, output_filepath)

        return EmbeddingOperationResult(
            success=True,
            message=(
                f"Successfully updated {student_name} ({roll_number}) with "
                f"{len(embeddings)} accepted image(s)"
            ),
            roll_number=roll_number,
            student_name=student_name,
        )

    except Exception as exc:
        logger.error(
            "update_existing_student [%s]: %s (%s)",
            roll_number,
            exc,
            type(exc).__name__,
        )
        return EmbeddingOperationResult(
            success=False,
            message=str(exc),
            roll_number=roll_number,
        )


# ---------------------------------------------------------------------------
# Task 7.5 — remove_student
# Requirements: 13.10, 13.12
# ---------------------------------------------------------------------------

def remove_student(
    roll_number: str,
    output_filepath: str = EMBEDDINGS_FILE,
) -> EmbeddingOperationResult:
    """
    Remove a student's record from the embedding store.

    If the student does not exist, returns a success result noting that the
    record was not found (no-op, consistent with
    :meth:`EmbeddingStore.delete_student` semantics).

    Parameters
    ----------
    roll_number : str
        The roll number of the student to remove.
    output_filepath : str
        Path to the pickle store.
        Defaults to :data:`EMBEDDINGS_FILE`.

    Returns
    -------
    EmbeddingOperationResult
        ``success=True`` always unless a storage-level exception occurs
        (``success=False``).

    Requirements: 13.10, 13.12
    """
    try:
        # Check whether the record exists so we can tailor the message
        existing = EmbeddingStore.get_student(roll_number, output_filepath)

        EmbeddingStore.delete_student(roll_number, output_filepath)

        if existing is not None:
            message = (
                f"Student record for roll_number '{roll_number}' "
                f"({existing.name}) successfully deleted"
            )
            student_name = existing.name
        else:
            message = (
                f"Student record for roll_number '{roll_number}' "
                f"was not found in store — no action taken"
            )
            student_name = None

        return EmbeddingOperationResult(
            success=True,
            message=message,
            roll_number=roll_number,
            student_name=student_name,
        )

    except Exception as exc:
        logger.error(
            "remove_student [%s]: %s (%s)",
            roll_number,
            exc,
            type(exc).__name__,
        )
        return EmbeddingOperationResult(
            success=False,
            message=str(exc),
            roll_number=roll_number,
        )
