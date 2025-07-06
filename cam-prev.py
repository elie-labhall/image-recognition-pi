#!/usr/bin/env python3
import cv2
import time
from picamera2 import Picamera2

# Configurable resolution and FPS
CAM_WIDTH, CAM_HEIGHT = 320, 140
TARGET_FPS = 40
FRAME_DURATION = 1.0 / TARGET_FPS

# Initialize camera
picam2 = Picamera2()

video_cfg = picam2.create_video_configuration(
    main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
    buffer_count=4
)
picam2.configure(video_cfg)
picam2.set_controls({"FrameRate": TARGET_FPS})  # Request 60 FPS (will clamp to sensor limits)
picam2.start()

print(f"Camera started at {CAM_WIDTH}x{CAM_HEIGHT}, target FPS: {TARGET_FPS}")

try:
    while True:
        frame_start = time.perf_counter()

        frame = picam2.capture_array()
        if frame is None:
            continue

        # Rotate 90° counterclockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Pi Live Feed", frame)

        # Allow 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.perf_counter() - frame_start
        sleep_time = FRAME_DURATION - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        fps = 1.0 / max(elapsed, 1e-6)
        print(f"Resolution: {CAM_WIDTH}x{CAM_HEIGHT}, Display FPS: {fps:.2f}")

finally:
    cv2.destroyAllWindows()
    picam2.close()
