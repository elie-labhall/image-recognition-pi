#!/usr/bin/env python3
""" 
Pi pan-tilt tracker with face-priority & relay trigger

• Tries face first (SSD-ResNet10); if lost >0.6 s ⇒ body (MobileNet-SSD)
• 3-2-1 countdown → relay shot → 1 s “Target Eliminated”
• During countdown/eliminated: tolerate 2 empty frames before cancel
• Live MJPEG stream:  http://<pi-ip>:5000/
"""

# ─── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH, TILT_CH         = 0, 1
PAN_MIN, PAN_MAX        = 20, 160
TILT_MIN, TILT_MAX      = 50, 130
PAN_CENTER, TILT_CENTER = 90, 90
TILT_DIR                = -1           # flip sign if needed

SCAN_RANGE_PAN, SCAN_RANGE_TILT = 40, 30
SCAN_STEP, SCAN_DELAY, SCAN_WAVES = 1, 0.10, 4

CENTER_TOL_X, CENTER_TOL_Y = 8, 8
GAIN_PAN, GAIN_TILT        = 0.05, 0.05
LOST_TIMEOUT               = 2.0        # s → scan

CAM_W, CAM_H               = 300, 225

# Face detector
FACE_PROTO   = "deploy.prototxt"
FACE_MODEL   = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_CONF    = 0.5
MIN_FACE_PIX = 225             # ~20×20 px
FACE_HOLD_SEC= 0.1             # keep face mode this long after loss

# Body detector
PERSON_PROTO = "MobileNetSSD_deploy.prototxt"
PERSON_MODEL = "MobileNetSSD_deploy.caffemodel"
PERSON_CLASS = 15
PERSON_CONF  = 0.5
MIN_BODY_FRAC= 0.02            # discard <2 % frame

# Firing
RELAY_PIN            = 6
COUNTDOWN_SEC        = 3        # integer
COUNTDOWN_LOST_TOL   = 0        # frames tolerated during countdown
SHOT_DURATION        = 0.25
ELIM_DISPLAY_SEC     = 1.0

mode_dic = {
    "burst": {"COUNTDOWN_SEC": 5, "SHOT_DURATION": 0.7},
    "warning": {"COUNTDOWN_SEC": 3, "SHOT_DURATION": 0.25}
}

mode = "warning"
COUNTOWN_SEC = mode_dic[mode]["COUNTDOWN_SEC"]
SHOT_DURATION = mode_dic[mode]["SHOT_DURATION"]

# ────────────────────────────────────────────────────────────────────────

import math, time, cv2, numpy as np, board, busio, RPi.GPIO as GPIO
from pathlib import Path
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string

# ─── Servo setup ───────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50
def set_angle(ch:int, ang:float):
    ang = max(0, min(180, ang))
    pca.channels[ch].duty_cycle = int((500 + (ang/180)*1900)*65535/20000)
pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)

# ─── Camera ────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}, buffer_count=1))
picam2.start()
CX, CY = CAM_W//2, CAM_H//2

# ─── Load DNNs ─────────────────────────────────────────────────────────
for f in (FACE_PROTO, FACE_MODEL, PERSON_PROTO, PERSON_MODEL):
    if not Path(f).exists():
        raise FileNotFoundError(f"Missing {f}")
net_face   = cv2.dnn.readNetFromCaffe(FACE_PROTO,   FACE_MODEL)
net_person = cv2.dnn.readNetFromCaffe(PERSON_PROTO, PERSON_MODEL)
for n in (net_face, net_person):
    n.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    n.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ─── Relay setup ───────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM); GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)
def trigger_shot():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(SHOT_DURATION)
    GPIO.output(RELAY_PIN, GPIO.LOW)

# ─── Flask UI ──────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """<!doctype html><html><head><title>Pi Tracker</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain;"></body></html>"""
@app.route("/")
def index(): return render_template_string(HTML)

# ─── Runtime state ─────────────────────────────────────────────────────
scan_dir = 1
last_seen = 0.0

last_face_time  = 0.0
last_face_box   = None

countdown_start, shot_time = None, None
lost_frames = 0

last_conf, last_fx, last_fy = 0.0, CX, CY

# ─── Stream generator ──────────────────────────────────────────────────
def gen_frames():
    global pan, tilt, scan_dir, last_seen
    global last_face_time, last_face_box
    global countdown_start, shot_time, lost_frames
    global last_conf, last_fx, last_fy

    while True:
        t0 = time.perf_counter()
        frame = picam2.capture_array()

        # ---- 1) try FACE detection -----------------------------------
        blob_f = cv2.dnn.blobFromImage(frame, 1.0,
                                       (300,300),
                                       (104.0,177.0,123.0))
        net_face.setInput(blob_f)
        det_f = net_face.forward()

        best_face, best_fconf = None, 0.0
        for i in range(det_f.shape[2]):
            conf = float(det_f[0,0,i,2])
            if conf < FACE_CONF: continue       # face not fround
            x1,y1,x2,y2 = (det_f[0,0,i,3:7] *
                           np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
            if (x2-x1)*(y2-y1) < MIN_FACE_PIX: continue    # face too small
            if conf > best_fconf:
                best_face, best_fconf = (x1,y1,x2,y2), conf

        if best_face is not None:
            last_face_box, last_face_time = best_face, time.time()

        # Use face if we have a fresh one (< hold time)
        use_face = (time.time() - last_face_time) < FACE_HOLD_SEC and last_face_box
        tracking_mode = "face" if use_face else "none"
        best_box, best_conf = (last_face_box, best_fconf) if use_face else (None, 0.0)

        # ---- 2) BODY detection if no face ----------------------------
        if tracking_mode == "none":
            blob_p = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
            net_person.setInput(blob_p); det_p = net_person.forward()
            for i in range(det_p.shape[2]):
                conf = float(det_p[0,0,i,2]); cls = int(det_p[0,0,i,1])
                if conf < PERSON_CONF or cls != PERSON_CLASS: continue
                x1,y1,x2,y2 = (det_p[0,0,i,3:7] *
                               np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
                if (x2-x1)*(y2-y1) < MIN_BODY_FRAC*CAM_W*CAM_H: continue
                if conf > best_conf:
                    best_box, best_conf = (x1,y1,x2,y2), conf
            if best_box: tracking_mode = "body"

        detected = best_box is not None

        # ---- Track or scan -------------------------------------------
        if detected:
            lost_frames = 0
            x1,y1,x2,y2 = best_box
            fx, fy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # servo adjust
            err_x = fx - CX
            if abs(err_x) > CENTER_TOL_X:
                pan -= err_x * GAIN_PAN
                pan = max(PAN_MIN, min(PAN_MAX, pan)); set_angle(PAN_CH, pan)

            err_y = fy - CY
            if abs(err_y) > CENTER_TOL_Y:
                tilt += TILT_DIR * err_y * GAIN_TILT
                tilt = max(TILT_MIN, min(TILT_MAX, tilt)); set_angle(TILT_CH, tilt)

            last_seen, last_conf = time.time(), best_conf
            last_fx, last_fy = fx, fy
        else:
            lost_frames += 1
            # scanning after LOST_TIMEOUT
            if time.time()-last_seen > LOST_TIMEOUT:
                pan += SCAN_STEP*scan_dir
                if pan>PAN_MAX or pan<PAN_MIN:
                    scan_dir*=-1; pan += SCAN_STEP*scan_dir
                t_norm=(pan-PAN_CENTER)/SCAN_RANGE_PAN
                tilt=TILT_CENTER+SCAN_RANGE_TILT*math.sin(math.pi*SCAN_WAVES*t_norm)
                tilt=max(TILT_MIN,min(TILT_MAX,tilt))
                set_angle(PAN_CH,pan); set_angle(TILT_CH,tilt)
                time.sleep(SCAN_DELAY)

        # ---- Countdown / shot FSM with frame tolerance ---------------
        target_ok = detected or lost_frames <= COUNTDOWN_LOST_TOL
        if target_ok:
            if shot_time:
                if time.time()-shot_time >= ELIM_DISPLAY_SEC:
                    shot_time=None; countdown_start=time.time()
            elif countdown_start is None:
                countdown_start = time.time()
            elif time.time()-countdown_start >= COUNTDOWN_SEC:
                trigger_shot(); shot_time=time.time(); countdown_start=None
        else:
            countdown_start, shot_time = None, None

        # ---- Overlay text --------------------------------------------
        fps = 1.0 / (time.perf_counter()-t0)
        cv2.putText(frame,f"{fps:4.1f} fps",(5,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,f"Conf: {last_conf:.2f}",(5,35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Pos: ({last_fx},{last_fy})",(5,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        cv2.putText(frame,f"Mode: {tracking_mode}",(5,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        print(f"Tracking mode: {tracking_mode}, ")

        if shot_time:
            overlay="Target Eliminated"
        elif countdown_start:
            remaining=COUNTDOWN_SEC-int(time.time()-countdown_start)
            overlay=str(max(1,remaining))
        else:
            overlay="Searching for target"
        cv2.putText(frame,overlay,(5,95),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # crosshair
        cx,cy=last_fx,last_fy; size,th=20,2
        cv2.line(frame,(cx-size,cy),(cx-5,cy),(0,255,0),th)
        cv2.line(frame,(cx+5,cy),(cx+size,cy),(0,255,0),th)
        cv2.line(frame,(cx,cy-size),(cx,cy-5),(0,255,0),th)
        cv2.line(frame,(cx,cy+5),(cx,cy+size),(0,255,0),th)
        cv2.circle(frame,(cx,cy),8,(0,255,0),1)

        ret,jpeg=cv2.imencode(".jpg",frame)
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg.tobytes()+b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── Main ───────────────────────────────────────────────────────────────
if __name__=="__main__":
    try:
        app.run(host="0.0.0.0",port=5000,threaded=True)
    finally:
        for ch in (PAN_CH,TILT_CH): pca.channels[ch].duty_cycle=0
        pca.deinit()
        GPIO.output(RELAY_PIN,GPIO.LOW); GPIO.cleanup()
        picam2.close()
        print("Clean shutdown.")
