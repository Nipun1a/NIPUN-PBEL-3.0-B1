"""
config.py
Central place for tunable constants keeeping these in one file for the later so that it do not 
hinder the logic
"""

import os 

#students face data
DATASET_ROOT = os.getenv("DATASET_ROOT", "CollectedImages")

#total images
IMAGES_PER_STUDENT = 100

#face size to save the photo
FACE_SIZE = (160,160)

#mediapipe detection confidence
MIN_DETECTION_CONFIDENCE = 0.6

#blur amount lower means rejected
BLUR_THRESHOLD= 60.0

#minimum face dimensions required to accept a frame
MIN_FACE_SIZE = 60

#shift pose slightly between shots
CAPTURE_DELAY = 0.15

#pOSE instructions
POSE_INSTRUCTIONS = [
    "Look Straight",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Smile"
]
