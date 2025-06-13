#!/usr/bin/env python3
"""
camera_face_body_detection.py

Captures video from the Raspberry Pi camera, detects faces or bodies,
and draws rectangles around them in real time.

Requirements:
  • Raspberry Pi Camera Module v2/v3 (properly connected)
  • python3-picamera2
  • python3-opencv
  • haarcascade_frontalface_default.xml
  • haarcascade_fullbody.xml
    (both placed in the same directory as this script)
"""

from typing import List, Tuple
import cv2
import numpy as np
from picamera2 import Picamera2

def detect_objects(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
    scaleFactor: float,
    minNeighbors: int,
    minSize: Tuple[int,int]
) -> List[Tuple[int, int, int, int]]:
    return cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )

def main() -> None:
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    if face_cascade.empty() or body_cascade.empty():
        print("❌ Error: Could not load one or more Haar cascades.")
        return

    # Initialize PiCamera2
    picam2 = Picamera2()
    cfg = picam2.preview_configuration
    cfg.main.size = (960, 720)  # or (1920, 1080) if performance allows
    cfg.main.format = 'RGB888'
    picam2.configure("preview")
    picam2.start()

    print("🎥 Streaming... press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1) Try face detection
            faces = detect_objects(gray, face_cascade,
                                   scaleFactor=1.1,
                                   minNeighbors=5,
                                   minSize=(30, 30))

            if len(faces) > 0:
                # draw red rectangles for each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 2)
            else:
                # fallback: upper-body detection in blue
                bodies = body_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50)
                )
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                (255, 0, 0), 2)


            # Show result
            cv2.imshow("Face & Body Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.close()

if __name__ == "__main__":
    main()
