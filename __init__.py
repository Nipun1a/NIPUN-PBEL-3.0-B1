"""
Reusable face-dataset collection service for the AI Attendance
Monitoring System. Import DatasetCollector and StudentDetails to use
this from a backend API, a CLI script, or anywhere else.
 
Example:
    from face_dataset_service import DatasetCollector, StudentDetails
 
    collector = DatasetCollector()
    result = collector.collect(
        StudentDetails(name="Yug", roll_number="102", department="CSE")
    ) 
    
    #isse badme meh fastapi se backend bana dunga
    
"""

try:
    from .dataset_collector import DatasetCollector
    from .models import StudentDetails, CaptureResult
    from .face_detector import FaceDetector, FaceDetection
except ImportError:  # Allow direct script-style usage
    from dataset_collector import DatasetCollector
    from models import StudentDetails, CaptureResult
    from face_detector import FaceDetector, FaceDetection

__all__ =[
    "DatasetCollector",
    "StudentDetails",
    "CaptureResult",
    "FaceDetector",
    "FaceDetection",
]
