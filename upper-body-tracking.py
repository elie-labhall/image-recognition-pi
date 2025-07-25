#!/usr/bin/env python3
"""
Pi pan-tilt person-tracker with MJPEG streaming

• SG90/MG90S servos via PCA9685 (I²C)
• MobileNet-SSD (Caffe) person detector
• Keeps target centred to ±10 px
• Crosshair drawn at the target’s centre
• Live stream:  http://<pi-ip>:5000/
"""

# ─── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH, TILT_CH        = 0, 1        # PCA9685 channels
PAN_MIN, PAN_MAX       = 20, 160
TILT_MIN, TILT_MAX     = 50, 130
PAN_CENTER, TILT_CENTER = 90, 90
TILT_DIR               = -1          # change to +1 if tilt is reversed

SCAN_RANGE_PAN  = 40
SCAN_RANGE_TILT = 30
SCAN_STEP       = 1
SCAN_DELAY      = 0.10
SCAN_WAVES      = 4

CENTER_TOL_X, CENTER_TOL_Y = 10, 10   # ±px tolerated error
GAIN_PAN,  GAIN_TILT       = 0.6, 0.6
LOST_TIMEOUT               = 2.0      # s w/out target → scan

CAM_W, CAM_H      = 300, 225
CONF_THRESHOLD    = 0.5
MIN_AREA_FRAC     = 0.02               # discard <2 % frame
PERSON_CLASS_ID   = 15
PROTO_PATH        = "MobileNetSSD_deploy.prototxt"
MODEL_PATH        = "MobileNetSSD_deploy.caffemodel"
# ────────────────────────────────────────────────────────────────────────


# ─── Imports ────────────────────────────────────────────────────────────
import math, time, cv2, numpy as np, board, busio
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string

# ─── Servo helpers ──────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50

def set_angle(ch: int, angle: float) -> None:
    angle = max(0, min(180, angle))
    pca.channels[ch].duty_cycle = int((500 + (angle / 180) * 1900) * 65535 / 20000)

pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan)
set_angle(TILT_CH, tilt)

# ─── Camera ─────────────────────────────────────────────────────────────
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (CAM_W, CAM_H), "format": "RGB888"},
    buffer_count=1
)
picam2.configure(cfg); picam2.start()
CX, CY = CAM_W // 2, CAM_H // 2

# ─── Detector ───────────────────────────────────────────────────────────
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ─── Flask UI ───────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """
<!doctype html><html><head><title>Pi Person Stream</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}"
     style="width:100vw;height:100vh;object-fit:contain;"></body></html>"""

@app.route("/")
def index(): return render_template_string(HTML)

# ─── Tracking / streaming generator ────────────────────────────────────
scan_dir  = 1
last_seen = 0.0

def gen_frames():
    global pan, tilt, scan_dir, last_seen
    last_conf, last_fx, last_fy = 0.0, CX, CY
    while True:
        t0    = time.perf_counter()
        frame = picam2.capture_array()

        # ── Detect person ───────────────────────────────────────────────
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        det  = net.forward()

        best_box, best_conf, best_area = None, 0.0, 0
        for i in range(det.shape[2]):
            conf = float(det[0,0,i,2]); cls = int(det[0,0,i,1])
            if conf < CONF_THRESHOLD or cls != PERSON_CLASS_ID: continue
            x1,y1,x2,y2 = (det[0,0,i,3:7] *
                           np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
            area = (x2-x1)*(y2-y1)
            if area < MIN_AREA_FRAC * CAM_W * CAM_H: continue
            if area > best_area:
                best_box, best_conf, best_area = (x1,y1,x2,y2), conf, area

        detected = best_box is not None
        if detected:
            x1,y1,x2,y2 = best_box
            fx, fy      = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # Proportional servo control
            err_x = fx - CX
            if abs(err_x) > CENTER_TOL_X:
                pan -= err_x * GAIN_PAN
                pan = max(PAN_MIN, min(PAN_MAX, pan)); set_angle(PAN_CH, pan)

            err_y = fy - CY
            if abs(err_y) > CENTER_TOL_Y:
                tilt += TILT_DIR * err_y * GAIN_TILT
                tilt = max(TILT_MIN, min(TILT_MAX, tilt)); set_angle(TILT_CH, tilt)

            last_seen = time.time()
            last_conf, last_fx, last_fy = best_conf, fx, fy

        # Lost target → scan
        if not detected and time.time() - last_seen > LOST_TIMEOUT:
            pan += SCAN_STEP * scan_dir
            if pan > PAN_MAX or pan < PAN_MIN:
                scan_dir *= -1; pan += SCAN_STEP * scan_dir
            t_norm = (pan - PAN_CENTER) / SCAN_RANGE_PAN
            tilt   = TILT_CENTER + SCAN_RANGE_TILT * math.sin(math.pi*SCAN_WAVES*t_norm)
            tilt   = max(TILT_MIN, min(TILT_MAX, tilt))
            set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)
            time.sleep(SCAN_DELAY)

        # ── Overlay info ────────────────────────────────────────────────
        fps = 1.0 / (time.perf_counter() - t0)
        cv2.putText(frame,f"{fps:4.1f} fps",(5,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,f"Conf: {last_conf:.2f}",(5,35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Pos: ({last_fx},{last_fy})",(5,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        # --- Draw moving crosshair at target centre --------------------
        cross_cx, cross_cy = last_fx, last_fy
        size, thick = 20, 2
        color = (0,255,0)

        # horizontal arms
        cv2.line(frame,(cross_cx-size, cross_cy),(cross_cx-5, cross_cy),color,thick)
        cv2.line(frame,(cross_cx+5, cross_cy),(cross_cx+size, cross_cy),color,thick)
        # vertical arms
        cv2.line(frame,(cross_cx, cross_cy-size),(cross_cx, cross_cy-5),color,thick)
        cv2.line(frame,(cross_cx, cross_cy+5),(cross_cx, cross_cy+size),color,thick)
        cv2.circle(frame,(cross_cx, cross_cy),8,color,1)

        # ── Encode & stream ────────────────────────────────────────────
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
        print("Cleaning up servos …")
        for ch in (PAN_CH, TILT_CH): pca.channels[ch].duty_cycle = 0
        pca.deinit(); picam2.close()
