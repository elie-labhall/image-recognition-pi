#!/usr/bin/env python3
"""
camera_face_detection.py

Captures video from the Raspberry Pi camera, detects faces, and draws rectangles
around them in real time.

Requirements:
  • Raspberry Pi Camera Module v2/v3 (properly connected)
  • python3-picamera2
  • python3-opencv
  • haarcascade_frontalface_default.xml in the same directory
"""

from typing import List, Tuple
import cv2
import numpy as np
from picamera2 import Picamera2

def detect_faces(
    gray_frame: np.ndarray,
    cascade: cv2.CascadeClassifier
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a grayscale image.

    Args:
        gray_frame: Grayscale image from the camera.
        cascade:    Pre-loaded OpenCV CascadeClassifier.

    Returns:
        A list of rectangles, each (x, y, w, h) around a detected face.
    """
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,      # how much the image size is reduced at each image scale
        minNeighbors=5,       # higher => fewer detections but higher quality
        minSize=(30, 30)      # ignore smaller detections
    )
    return faces

def main() -> None:
    """Main routine: initialize camera, load model, and enter the capture loop."""
    # 1. Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: Could not load haarcascade_frontalface_default.xml")
        return

    # 2. Initialize the camera
    picam2 = Picamera2()
    # Configure for a 640×480 RGB preview
    config = picam2.preview_configuration
    config.main.size = (640, 480)
    config.main.format = 'RGB888'
    picam2.configure("preview")
    picam2.start()

    print("🎥 Streaming... press 'q' in the window to quit.")

    try:
        while True:
            # 3. Capture a single frame as a NumPy array
            frame: np.ndarray = picam2.capture_array()

            # 4. Convert to grayscale (face detector expects gray images)
            gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 5. Detect faces
            faces = detect_faces(gray, face_cascade)

            # 6. Draw a green rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (203, 192, 255), 2)

            # 7. Show the result in a window named “Face Detection”
            cv2.imshow("Face Detection", frame)

            # 8. If the user presses 'q', exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 9. Clean up windows and camera on exit
        cv2.destroyAllWindows()
        picam2.close()

if __name__ == "__main__":
    main()


