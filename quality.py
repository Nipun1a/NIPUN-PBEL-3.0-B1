'''
simple image_qualitychecks used to reject blurry or unusable face
'''

import cv2
import numpy as np

try:
    from .config import BLUR_THRESHOLD, MIN_FACE_SIZE
except ImportError:
    from config import BLUR_THRESHOLD, MIN_FACE_SIZE


def variance_of_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_blurry(image_bgr: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    # returns true if the image is too blurry to keep
    return variance_of_laplacian(image_bgr) < threshold

def is_face_too_small(face_width: int, face_height: int, min_size: int = MIN_FACE_SIZE) -> bool:
    return face_width < min_size or face_height < min_size
