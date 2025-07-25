#!/usr/bin/env python3
"""
Pi pan-tilt person-tracker with MJPEG streaming + relay trigger

• SG90/MG90S servos via PCA9685
• MobileNet-SSD person detector
• Countdown → shot → 1 s “Target Eliminated” overlay
• Automatic re-arming while the target remains
• Live feed at  http://<pi-ip>:5000/
"""

# ─── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH, TILT_CH        = 0, 1
PAN_MIN, PAN_MAX       = 20, 160
TILT_MIN, TILT_MAX     = 50, 130
PAN_CENTER, TILT_CENTER = 90, 90
TILT_DIR               = -1          # flip sign if tilting the wrong way

SCAN_RANGE_PAN  = 40
SCAN_RANGE_TILT = 30
SCAN_STEP       = 1
SCAN_DELAY      = 0.10
SCAN_WAVES      = 4

CENTER_TOL_X, CENTER_TOL_Y = 8, 8 # ±px tolerated error
GAIN_PAN,  GAIN_TILT       = 0.1, 0.1
LOST_TIMEOUT               = 2.0     # s no target → scan mode

CAM_W, CAM_H            = 300, 225
CONF_THRESHOLD          = 0.5
MIN_AREA_FRAC           = 0.02
PERSON_CLASS_ID         = 15
PROTO_PATH              = "MobileNetSSD_deploy.prototxt"
MODEL_PATH              = "MobileNetSSD_deploy.caffemodel"

# Relay / firing
RELAY_PIN               = 6
COUNTDOWN_SEC           = 3          # must be an integer
SHOT_DURATION           = 0.25       # seconds HIGH
ELIM_DISPLAY_SEC        = 1.0        # “Target Eliminated” display time
# ────────────────────────────────────────────────────────────────────────

import math, time, cv2, numpy as np, board, busio
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string

# ─── Servo helpers ──────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50
def set_angle(ch:int, ang:float):
    ang = max(0, min(180, ang))
    pca.channels[ch].duty_cycle = int((500 + (ang/180)*1900)*65535/20000)

pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)

# ─── Camera ─────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main = {"size": (CAM_W, CAM_H), "format": "RGB888"},
        buffer_count = 1))
picam2.start()
CX, CY = CAM_W//2, CAM_H//2

# ─── Detector ───────────────────────────────────────────────────────────
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ─── Relay setup ────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)
def trigger_shot():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(SHOT_DURATION)
    GPIO.output(RELAY_PIN, GPIO.LOW)

# ─── Flask UI ───────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """
<!doctype html><html><head><title>Pi Person Stream</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain;">
</body></html>"""
@app.route("/")                # root page
def index(): return render_template_string(HTML)

# ─── Runtime state ─────────────────────────────────────────────────────
scan_dir = 1
last_seen = 0.0

countdown_start = None         # None => not counting
shot_time       = None         # time when shot fired
last_conf = 0.0
last_fx, last_fy = CX, CY

# ─── Stream generator ──────────────────────────────────────────────────
def gen_frames():
    global pan, tilt, scan_dir, last_seen
    global countdown_start, shot_time, last_conf, last_fx, last_fy

    while True:
        frame_t0 = time.perf_counter()
        frame = picam2.capture_array()

        # ─── Detect person ────────────────────────────────────────────
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob); det = net.forward()

        best_box = None; best_conf = 0.0; best_area = 0
        for i in range(det.shape[2]):
            conf = float(det[0,0,i,2]); cls = int(det[0,0,i,1])
            if conf < CONF_THRESHOLD or cls != PERSON_CLASS_ID: continue
            x1,y1,x2,y2 = (det[0,0,i,3:7] * np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
            area = (x2-x1)*(y2-y1)
            if area < MIN_AREA_FRAC*CAM_W*CAM_H: continue
            if area > best_area: best_box, best_conf, best_area = (x1,y1,x2,y2), conf, area

        detected = best_box is not None
        if detected:
            x1,y1,x2,y2 = best_box
            fx, fy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # ── Track ────────────────────────────────────────────────
            err_x = fx - CX
            if abs(err_x) > CENTER_TOL_X:
                pan -= err_x * GAIN_PAN
                pan = max(PAN_MIN, min(PAN_MAX, pan))
                set_angle(PAN_CH, pan)

            err_y = fy - CY
            if abs(err_y) > CENTER_TOL_Y:
                tilt += TILT_DIR * err_y * GAIN_TILT
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
                set_angle(TILT_CH, tilt)

            last_seen  = time.time()
            last_conf  = best_conf
            last_fx, last_fy = fx, fy

            # ── Countdown / shot FSM ────────────────────────────────
            if shot_time:                                 # in “Eliminated” period
                if time.time() - shot_time >= ELIM_DISPLAY_SEC:
                    shot_time = None                      # elapsed → re-arm
                    countdown_start = time.time()
            elif countdown_start is None:                 # fresh target
                countdown_start = time.time()
            elif time.time() - countdown_start >= COUNTDOWN_SEC:
                # countdown finished → fire
                trigger_shot()
                shot_time = time.time()
                countdown_start = None

        else:
            # target lost → reset state
            countdown_start = None
            shot_time = None

            # scanning after LOST_TIMEOUT
            if time.time() - last_seen > LOST_TIMEOUT:
                pan += SCAN_STEP * scan_dir
                if pan > PAN_MAX or pan < PAN_MIN:
                    scan_dir *= -1; pan += SCAN_STEP * scan_dir
                t_norm = (pan - PAN_CENTER) / SCAN_RANGE_PAN
                tilt = TILT_CENTER + SCAN_RANGE_TILT * math.sin(math.pi*SCAN_WAVES*t_norm)
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
                set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)
                time.sleep(SCAN_DELAY)

        # ─── Overlays ────────────────────────────────────────────────
        fps = 1.0 / (time.perf_counter() - frame_t0)
        cv2.putText(frame, f"{fps:4.1f} fps", (5,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame, f"Conf: {last_conf:.2f}", (5,35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame, f"Pos: ({last_fx},{last_fy})", (5,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        overlay_txt = None
        if shot_time:
            overlay_txt = "Target Eliminated"
        elif countdown_start:
            remaining = COUNTDOWN_SEC - int(time.time() - countdown_start)
            overlay_txt = str(max(1, remaining))
        else:
            overlay_txt = "Searching for target"

        cv2.putText(frame, overlay_txt, (5,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)



        # crosshair
        cx, cy = last_fx, last_fy
        size, th = 20, 2
        cv2.line(frame,(cx-size,cy),(cx-5,cy),(0,255,0),th)
        cv2.line(frame,(cx+5,cy),(cx+size,cy),(0,255,0),th)
        cv2.line(frame,(cx,cy-size),(cx,cy-5),(0,255,0),th)
        cv2.line(frame,(cx,cy+5),(cx,cy+size),(0,255,0),th)
        cv2.circle(frame,(cx,cy),8,(0,255,0),1)

        # ─── Stream ─────────────────────────────────────────────────
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
        for ch in (PAN_CH, TILT_CH): pca.channels[ch].duty_cycle = 0
        pca.deinit()
        GPIO.output(RELAY_PIN, GPIO.LOW)
        GPIO.cleanup()
        picam2.close()
        print("Clean shutdown.")
