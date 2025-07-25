#!/usr/bin/env python3
"""
Pi pan-tilt FACE-tracker with MJPEG streaming (Raspberry Pi 5 + Camera Module 3)

• SG90 / MG90S servos driven by PCA9685 (I²C, 50 Hz)
• Face detection: SSD-ResNet10 (OpenCV DNN, Caffe model)
• Tracks faces as small as ~20×20 px (≈ 7–8 m)
• Crosshair drawn at the face centre
• Stream at http://<pi-ip>:5000/
"""

# ─── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH, TILT_CH         = 0, 1
PAN_MIN, PAN_MAX        = 20, 160
TILT_MIN, TILT_MAX      = 50, 130
PAN_CENTER, TILT_CENTER = 90, 90
TILT_DIR                = -1         # flip to +1 if tilt is reversed

SCAN_RANGE_PAN  = 40                 # scanning when no face
SCAN_RANGE_TILT = 30
SCAN_STEP       = 1
SCAN_DELAY      = 0.10
LOST_TIMEOUT    = 2.0                # seconds without face → scan

CENTER_TOL_X, CENTER_TOL_Y = 40, 40  # tolerated pixel error
GAIN_PAN,  GAIN_TILT       = 0.55, 0.55
MAX_STEP                   = 2        # °/frame (limits overshoot)

CAM_W, CAM_H   = 320, 440            # capture resolution (can raise to 1280×720)
MIN_PIXELS     = 400                 # keep detections ≥400 px² (~20×20 px)
DNN_CONF_THR   = 0.5                 # min detection confidence
PROTO_PATH     = "deploy.prototxt"
MODEL_PATH     = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# ─── Imports ────────────────────────────────────────────────────────────
import math, time
from pathlib import Path
import cv2, numpy as np
import board, busio
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string

# ─── Servo setup ───────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50

def set_angle(ch: int, angle: float) -> None:
    """Convert angle (0–180°) to a 500–2400 µs PWM pulse."""
    angle = max(0, min(180, angle))
    pulse = 500 + (angle / 180) * 1900
    pca.channels[ch].duty_cycle = int(pulse * 65535 / 20_000)

pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)

# ─── Camera ─────────────────────────────────────────────────────────────
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (CAM_W, CAM_H), "format": "RGB888"},
    buffer_count=1
)
picam2.configure(cfg); picam2.start()
CX, CY = CAM_W // 2, CAM_H // 2

# ─── Load DNN face detector ────────────────────────────────────────────
for f in (PROTO_PATH, MODEL_PATH):
    if not Path(f).exists():
        raise FileNotFoundError(f"Missing {f} – place it beside this script.")

net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ─── Flask UI ──────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """<!doctype html><html><head><title>Pi Face Stream</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}"
     style="width:100vw;height:100vh;object-fit:contain;"></body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

# ─── Tracking / streaming generator ────────────────────────────────────
scan_dir, last_seen = 1, 0.0
last_fx, last_fy    = CX, CY

def gen_frames():
    global pan, tilt, scan_dir, last_seen, last_fx, last_fy
    while True:
        t0    = time.perf_counter()
        frame = picam2.capture_array()

        # --- Run DNN face detection (downsized to 300×300) -------------
        blob = cv2.dnn.blobFromImage(frame, 1.0,
                                     (300, 300),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        dets = net.forward()

        detected, best_conf, best_box = False, 0.0, None
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < DNN_CONF_THR: continue
            x1, y1, x2, y2 = (
                dets[0, 0, i, 3:7] * np.array([CAM_W, CAM_H, CAM_W, CAM_H])
            ).astype(int)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_PIXELS:  # constant pixel threshold
                continue
            if conf > best_conf:
                best_conf, best_box = conf, (x1, y1, x2, y2)

        if best_box is not None:
            detected = True
            x1, y1, x2, y2 = best_box
            fx, fy = x1 + (x2 - x1)//2, y1 + (y2 - y1)//2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Servo correction --------------------------------------
            err_x = fx - CX
            if abs(err_x) > CENTER_TOL_X:
                delta = max(-MAX_STEP, min(MAX_STEP, -err_x * GAIN_PAN))
                pan  += delta
                pan   = max(PAN_MIN, min(PAN_MAX, pan))
                set_angle(PAN_CH, pan)

            err_y = fy - CY
            if abs(err_y) > CENTER_TOL_Y:
                delta = max(-MAX_STEP, min(MAX_STEP, TILT_DIR * err_y * GAIN_TILT))
                tilt += delta
                tilt  = max(TILT_MIN, min(TILT_MAX, tilt))
                set_angle(TILT_CH, tilt)

            last_seen, last_fx, last_fy = time.time(), fx, fy

        # --- If no face recently, sweep in a scan pattern --------------
        if not detected and time.time() - last_seen > LOST_TIMEOUT:
            pan += SCAN_STEP * scan_dir
            if pan > PAN_MAX or pan < PAN_MIN:
                scan_dir *= -1; pan += SCAN_STEP * scan_dir
            t_norm = (pan - PAN_CENTER) / SCAN_RANGE_PAN
            tilt   = TILT_CENTER + SCAN_RANGE_TILT * math.sin(math.pi * 4 * t_norm)
            tilt   = max(TILT_MIN, min(TILT_MAX, tilt))
            set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)
            time.sleep(SCAN_DELAY)

        # --- Draw FPS and crosshair -----------------------------------
        fps = 1.0 / (time.perf_counter() - t0)
        cv2.putText(frame, f"{fps:4.1f} fps", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "DNN face", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"Pos: ({last_fx},{last_fy})", (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        size, thick, color = 20, 2, (0, 255, 0)
        cv2.line(frame, (last_fx - size, last_fy), (last_fx - 5, last_fy), color, thick)
        cv2.line(frame, (last_fx + 5, last_fy), (last_fx + size, last_fy), color, thick)
        cv2.line(frame, (last_fx, last_fy - size), (last_fx, last_fy - 5), color, thick)
        cv2.line(frame, (last_fx, last_fy + 5), (last_fx, last_fy + size), color, thick)
        cv2.circle(frame, (last_fx, last_fy), 8, color, 1)

        # --- JPEG encode & stream -------------------------------------
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        print("Cleaning up servos…")
        for ch in (PAN_CH, TILT_CH): pca.channels[ch].duty_cycle = 0
        pca.deinit(); picam2.close()
