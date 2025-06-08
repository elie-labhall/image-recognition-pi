#!/usr/bin/env python3
"""
Diagonal-scan face tracker for Raspberry Pi.

• Pan-tilt servos (SG90) via PCA9685
• Diagonal raster scan around centre until face appears
• Keeps face centred using simple proportional control
• Resumes scanning if face lost
"""
import math

# ─── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH,  TILT_CH      = 0, 1      # PCA9685 channels (pan, tilt)

# Mechanical limits (to avoid collisions)
PAN_MIN,  PAN_MAX     = 30, 150
TILT_MIN, TILT_MAX    = 50, 130

# Start at these angles
PAN_CENTER            = 90
TILT_CENTER           = 90

# Scan only half the range around centre
SCAN_RANGE_PAN        = 40         # ±40° from PAN_CENTER
SCAN_RANGE_TILT       = 30         # ±30° from TILT_CENTER
SCAN_STEP             = 1          # degrees per scan step
SCAN_DELAY            = 0.1       # seconds per scan tick
SCAN_WAVES            = 4     # how many up/down waves per sweep


# Face-tracking parameters
CENTER_TOL_X          = 40         # pixels
CENTER_TOL_Y          = 30
GAIN_PAN              = 0.03       # deg / pixel error
GAIN_TILT             = 0.03
LOST_TIMEOUT          = 2.0        # seconds without face before scanning
# ────────────────────────────────────────────────────────────────────────

import time, cv2, numpy as np
import board, busio
from adafruit_pca9685 import PCA9685
from picamera2 import Picamera2

# ─── Servo helper ───────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50

def set_angle(ch: int, angle: float):
    """Move servo to given angle, clamped to 0–180°."""
    angle = max(0, min(180, angle))
    pulse_us = 500 + (angle / 180) * 1900          # 0°→500 µs, 180°→2400 µs
    duty = int(pulse_us * 65535 / 20000)           # 20 ms @ 50 Hz
    pca.channels[ch].duty_cycle = duty

pan  = PAN_CENTER
tilt = TILT_CENTER
set_angle(PAN_CH,  pan)
set_angle(TILT_CH, tilt)

# ─── Camera + face detector ────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

picam2 = Picamera2()
cfg = picam2.preview_configuration; cfg.main.size = (640, 480); cfg.main.format = 'RGB888'
picam2.configure("preview"); picam2.start()
W, H = cfg.main.size
CX, CY = W // 2, H // 2

# ─── Scan state ────────────────────────────────────────────────────────
scan_dir = 1   # 1=pan right, -1=pan left
last_face_time = 0

print("Running…  press  q  (video window)  or  Ctrl+C  to quit.")
try:
    while True:
        frame = picam2.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces):                               # —— FACE FOUND ——
            x, y, w, h = faces[0]
            fx, fy = x + w//2, y + h//2

            # Pan error
            err_x = fx - CX
            if abs(err_x) > CENTER_TOL_X:
                pan -= err_x * GAIN_PAN
                pan = max(PAN_MIN, min(PAN_MAX, pan))
                set_angle(PAN_CH, pan)

            # Tilt error
            err_y = fy - CY
            if abs(err_y) > CENTER_TOL_Y:
                tilt += err_y * GAIN_TILT
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
                set_angle(TILT_CH, tilt)

            last_face_time = time.time()

            print(f"🎯 Tracking face at ({fx}, {fy}) | Pan: {pan:.1f} | Tilt: {tilt:.1f}")


        else:                                        # —— NO FACE ——
            if time.time() - last_face_time > LOST_TIMEOUT:
                # Diagonal raster scan around centre
                # advance pan
                pan += SCAN_STEP * scan_dir
                if pan > PAN_MAX or pan < PAN_MIN:
                    scan_dir *= -1
                    pan += SCAN_STEP * scan_dir

                # compute tilt as sine wave of pan
                t = (pan - PAN_CENTER) / SCAN_RANGE_PAN  # normalize: −1 to +1
                tilt = TILT_CENTER + SCAN_RANGE_TILT * math.sin(math.pi * SCAN_WAVES * t)

                # clamp & set angles
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
                set_angle(PAN_CH, pan)
                set_angle(TILT_CH, tilt)

                time.sleep(SCAN_DELAY)

                print(f"🔍 Scanning... Pan: {pan:.1f} | Tilt: {tilt:.1f}")


        # Draw rectangles for feedback
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (232, 72, 152), 2)
        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Cleaning up…")
    for ch in (PAN_CH, TILT_CH):
        pca.channels[ch].duty_cycle = 0
    pca.deinit()
    cv2.destroyAllWindows()
    picam2.close()
    print("Exited.")
