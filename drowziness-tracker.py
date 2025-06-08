#!/usr/bin/env python3
"""
camera_face_detection_sleep_ear_independent.py

Detects faces and eye closures using MediaPipe Face Mesh. Computes EAR (Eye Aspect Ratio)
separately for each eye. If both eyes remain fairly closed for a threshold number of frames,
displays "WAKE UP!" on the video feed.
"""


from typing import List, Tuple
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

import lgpio
import time


def detect_faces(
    gray_frame: np.ndarray,
    cascade: cv2.CascadeClassifier
) -> List[Tuple[int, int, int, int]]:
    return cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

def compute_ear(landmarks, eye_indices, w: int, h: int) -> float:
    """Compute Eye Aspect Ratio (EAR) for one eye."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def main() -> None:
    # Load face Haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: Could not load haarcascade_frontalface_default.xml")
        return

    # Initialize MediaPipe Face Mesh
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # EAR config
    closed_eyes_frames = 0
    CLOSED_FRAMES_THRESHOLD = 30      # ~1 second at 30 FPS
    EAR_THRESHOLD = 0.26              # more sensitive (light drowsiness)

    # Eye landmark indices (MediaPipe)
    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [263, 387, 385, 362, 380, 373]

    chip = lgpio.gpiochip_open(0)
    BUZZER_PIN = 17
    lgpio.gpio_claim_output(chip, BUZZER_PIN)




    # Start camera
    picam2 = Picamera2()
    config = picam2.preview_configuration
    config.main.size = (640, 480)
    config.main.format = 'RGB888'
    picam2.configure("preview")
    picam2.start()

    print("✅ Streaming... press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection (optional visualization)
            faces = detect_faces(gray, face_cascade)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (232, 72, 152), 2)

            # MediaPipe landmark detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                H, W, _ = frame.shape

                # Compute EAR separately
                left_ear = compute_ear(lm.landmark, LEFT, W, H)
                right_ear = compute_ear(lm.landmark, RIGHT, W, H)

                # If both eyes are "fairly closed", count it as a drowsy frame
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    closed_eyes_frames += 1
                else:
                    closed_eyes_frames = 0

                # (Optional) Display EARs for debugging
                cv2.putText(frame, f"EAR-L: {left_ear:.2f}", (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"EAR-R: {right_ear:.2f}", (120, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Trigger WAKE UP if both eyes closed for too long
            if closed_eyes_frames >= CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "WAKE UP!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                lgpio.gpio_write(chip, BUZZER_PIN, 1)
                time.sleep(0.05)
                lgpio.gpio_write(chip, BUZZER_PIN, 0)



            # Show output
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.close()
        lgpio.gpiochip_close(chip)



if __name__ == "__main__":
    main()
