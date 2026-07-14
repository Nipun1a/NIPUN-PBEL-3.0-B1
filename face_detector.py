'''
face detector with the help of mediapipe 
detects faces in BGR frame
return bounding boxes in pixel coordinates
cropping a detected face out of the frame

'''

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - dependency may be unavailable in some environments
    mp = None


@dataclass
class FaceDetection:
    #repersent a single detected face in pixel coordinates
    x: int
    y: int 
    width: int 
    height: int 
    confidence: float 
    
    @property
    def box(self) -> Tuple[int , int , int ,int ]:
        return self.x,self.y,self.width,self.height


class FaceDetector:
    def __init__(self,min_detection_confidence: float = 0.6,model_selection: int=0):
        self.detector = None
        self._face_cascade = None

        if mp is not None:
            try:
                self._mp_face_detection = mp.solutions.face_detection
                self.detector = self._mp_face_detection.FaceDetection(
                    min_detection_confidence=min_detection_confidence,
                    model_selection=model_selection,
                )
            except (AttributeError, ImportError):
                self.detector = None

        if self.detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError("Unable to initialize face detector. Please install mediapipe or ensure OpenCV data files are available.")
    
    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        frame_h , frame_w = frame_bgr.shape[:2]
        detections:List[FaceDetection]=[]

        if self.detector is not None:
            results = self.detector.process(frame_bgr)
            if not results.detections:
                return detections

            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                
                x= int(bbox.xmin * frame_w)
                y= int(bbox.ymin * frame_h)
                w = int(bbox.width * frame_w)
                h = int(bbox.height * frame_h)
                
                #clamp to frame bounds mediapipe can return slightly out of frame boxes
                
                x = max(0,x)
                y = max(0,y)
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                detections.append(
                    FaceDetection(
                        x=x, y=y,width=w,height=h,
                        confidence=det.score[0] if det.score else 0.0,
                        )
                )
            return detections

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            detections.append(FaceDetection(x=x, y=y, width=w, height=h, confidence=0.0))

        return detections
    
    @staticmethod
    def crop_face(frame_bgr: np.ndarray, detection: FaceDetection, margin: float= 0.15) -> np.ndarray:
        
        frame_h , frame_w = frame_bgr.shape[:2]
        x,y,w,h = detection.box
        
        mx = int(w*margin)
        my = int(h*margin)
        
        x1 = max(0, x-mx)
        y1 = max(0, y-my)
        x2 = min(frame_w,x+w+mx)
        y2 = min(frame_h,y+h+my)
        
        return frame_bgr[y1:y2, x1:x2]
    
    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        