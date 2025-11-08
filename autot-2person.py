#!/usr/bin/env python3
"""
Pi dual-target tracker (FACE-ONLY) with LED trigger + logging + overlays

• Detects and tracks up to two faces using SSD-ResNet10
• Two highest-confidence boxes above threshold:
      PRIMARY = rightmost, SECONDARY = leftmost
• Keeps each target for 2 s after loss to prevent flicker
• Logs detections, servo moves, and simulated firing
• Blue box = primary, Green box = secondary
• Stream at http://<pi-ip>:5000/video_feed
"""

print("started")

# ── CONFIG ─────────────────────────────────────────────────────────────
PAN_CH, TILT_CH          = 0, 1
PAN_MIN, PAN_MAX         = 30, 150
TILT_MIN, TILT_MAX       = 60, 120
PAN_CENTER, TILT_CENTER  = 90, 90
TILT_DIR                 = -1

CENTER_TOL_X, CENTER_TOL_Y = 12, 12
GAIN_PAN, GAIN_TILT       = 0.10, 0.07
LOST_TIMEOUT               = 2.0
TARGET_PERSIST_SEC         = 2.0

CAM_W, CAM_H               = 300, 225

SCAN_RANGE_PAN , SCAN_RANGE_TILT = 40, 30
SCAN_STEP, SCAN_WAVES, SCAN_DELAY = 1, 4, 0.10
scan_dir = 1

# Face detector
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_CONF  = 0.8
MIN_FACE_PIX = 225

# LED trigger
LED_PIN             = 6
COUNTDOWN_SEC       = 3
SHOT_DURATION       = 0.5
SECOND_DELAY        = 0.4
ELIM_DISPLAY_SEC    = 1.0

# Colors
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLW = (0, 255, 255)
RED   = (0, 0, 255)

# ── IMPORTS ────────────────────────────────────────────────────────────
import math, time, threading, cv2, numpy as np, board, busio, RPi.GPIO as GPIO
from pathlib import Path
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string
from datetime import datetime

# ── LOGGER ─────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── SERVO INIT ─────────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50
def set_angle(ch:int, deg:float):
    deg = max(0,min(180,deg))
    pca.channels[ch].duty_cycle = int((500 + (deg/180)*1900)*65535/20000)
pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)
log(f"Servos initialized → pan={pan}, tilt={tilt}")

# ── CAMERA ─────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size":(CAM_W,CAM_H),"format":"RGB888"}, buffer_count=1))
picam2.start()
CX, CY = CAM_W//2, CAM_H//2
log("Camera started")

# ── LOAD DNN (FACE) ───────────────────────────────────────────────────
for f in (FACE_PROTO, FACE_MODEL):
    if not Path(f).exists(): raise FileNotFoundError(f"Missing {f}")
net_face = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
net_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
log("Loaded face detection model")

# ── GPIO / LED ─────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
def fire():
    log("⚡ LED ON (simulated fire)")
    GPIO.output(LED_PIN, GPIO.HIGH); time.sleep(SHOT_DURATION)
    GPIO.output(LED_PIN, GPIO.LOW)

# ── FLASK UI ──────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """<!doctype html><html><head><title>Dual Face Tracker</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain;">
</body></html>"""
@app.route("/")
def index(): return render_template_string(HTML)

# ── STATE VARS ─────────────────────────────────────────────────────────
primary_bbox = secondary_bbox = None
primary_center = secondary_center = None
primary_conf = secondary_conf = 0.0
primary_last_seen = secondary_last_seen = 0.0
secondary_locked_center = None

countdown_start = shot_time = None
pending_second = False
second_start_t = 0.0
last_fx, last_fy, last_conf = CX, CY, 0.0
lost_frames = 0
last_seen = 0.0

# ── DETECTION ─────────────────────────────────────────────────────────
def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
    net_face.setInput(blob)
    det = net_face.forward()
    out=[]
    for i in range(det.shape[2]):
        conf=float(det[0,0,i,2])
        if conf<FACE_CONF: continue
        x1,y1,x2,y2=(det[0,0,i,3:7]*np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
        if (x2-x1)*(y2-y1)<MIN_FACE_PIX: continue
        cx,cy=(x1+x2)//2,(y1+y2)//2
        out.append({"conf":conf,"bbox":(x1,y1,x2,y2),"center":(cx,cy)})
    out.sort(key=lambda d:d["conf"],reverse=True)
    return out[:2]

# ── STREAM GENERATOR ──────────────────────────────────────────────────
def gen_frames():
    global pan, tilt, scan_dir
    global primary_bbox, primary_center, primary_conf, primary_last_seen
    global secondary_bbox, secondary_center, secondary_conf, secondary_last_seen
    global countdown_start, shot_time, pending_second, second_start_t, secondary_locked_center
    global last_fx, last_fy, last_conf, lost_frames, last_seen

    log("Video feed generator started")

    while True:
        t0=time.perf_counter()
        frame=picam2.capture_array()
        detections=detect_faces(frame)
        now_t=time.time()

        # Choose up to two detections, order by x
        if len(detections)==2 and detections[0]["center"][0]<detections[1]["center"][0]:
            detections[0],detections[1]=detections[1],detections[0]
        obs_primary=detections[0] if len(detections)>=1 else None
        obs_secondary=detections[1] if len(detections)>=2 else None

        if obs_primary:
            primary_bbox=obs_primary["bbox"]; primary_center=obs_primary["center"]
            primary_conf=obs_primary["conf"]; primary_last_seen=now_t
        if obs_secondary:
            secondary_bbox=obs_secondary["bbox"]; secondary_center=obs_secondary["center"]
            secondary_conf=obs_secondary["conf"]; secondary_last_seen=now_t

        primary_active=(now_t-primary_last_seen)<=TARGET_PERSIST_SEC and primary_bbox is not None
        secondary_active=(now_t-secondary_last_seen)<=TARGET_PERSIST_SEC and secondary_bbox is not None
        any_active=primary_active or secondary_active

        # Aim logic
        aim_center=None
        if pending_second and secondary_locked_center:
            aim_center=secondary_locked_center
        elif primary_active:
            aim_center=primary_center
        elif secondary_active:
            aim_center=secondary_center

        if aim_center:
            fx,fy=aim_center
            dx,dy=fx-CX,fy-CY
            moved=False
            if abs(dx)>CENTER_TOL_X:
                pan-=dx*GAIN_PAN; pan=max(PAN_MIN,min(PAN_MAX,pan)); set_angle(PAN_CH,pan); moved=True
            if abs(dy)>CENTER_TOL_Y:
                tilt+=TILT_DIR*dy*GAIN_TILT; tilt=max(TILT_MIN,min(TILT_MAX,tilt)); set_angle(TILT_CH,tilt); moved=True
            if moved:
                log(f"Target ({fx},{fy}) conf={primary_conf:.2f} → pan={pan:.1f}, tilt={tilt:.1f}")
            last_fx,last_fy,last_conf=fx,fy,primary_conf
            last_seen=now_t
            lost_frames=0
        else:
            lost_frames+=1

        # Firing logic
        if pending_second and now_t-second_start_t>=SECOND_DELAY:
            log("Second target firing"); fire(); pending_second=False; shot_time=now_t; secondary_locked_center=None
        elif any_active:
            if shot_time and now_t-shot_time>=ELIM_DISPLAY_SEC:
                shot_time=None; countdown_start=now_t
            elif countdown_start is None:
                countdown_start=now_t
            elif now_t-countdown_start>=COUNTDOWN_SEC and primary_active:
                log("Primary target firing")
                fire(); shot_time=now_t; countdown_start=None
                if secondary_active or (now_t-secondary_last_seen)<=TARGET_PERSIST_SEC:
                    pending_second=True; second_start_t=now_t
                    secondary_locked_center=secondary_center or (
                        ((secondary_bbox[0]+secondary_bbox[2])//2,
                         (secondary_bbox[1]+secondary_bbox[3])//2) if secondary_bbox else None)
                    if secondary_locked_center:
                        sx,sy=secondary_locked_center
                        pan-= (sx-CX)*GAIN_PAN; tilt+=TILT_DIR*(sy-CY)*GAIN_TILT
                        pan=max(PAN_MIN,min(PAN_MAX,pan)); tilt=max(TILT_MIN,min(TILT_MAX,tilt))
                        set_angle(PAN_CH,pan); set_angle(TILT_CH,tilt)
                        log(f"Re-aimed for secondary target @ {secondary_locked_center}")
        else:
            countdown_start=None; shot_time=None; pending_second=False

        # Scan if idle
        if not any_active and now_t-last_seen>LOST_TIMEOUT:
            pan+=SCAN_STEP*scan_dir
            if pan>PAN_MAX or pan<PAN_MIN:
                scan_dir*=-1; pan+=SCAN_STEP*scan_dir
            t_norm=(pan-PAN_CENTER)/SCAN_RANGE_PAN
            tilt=TILT_CENTER+SCAN_RANGE_TILT*math.sin(math.pi*SCAN_WAVES*t_norm)
            tilt=max(TILT_MIN,min(TILT_MAX,tilt))
            set_angle(PAN_CH,pan); set_angle(TILT_CH,tilt)
            log(f"Scanning... pan={pan:.1f}, tilt={tilt:.1f}")
            time.sleep(SCAN_DELAY)

        # Overlays
        fps=1.0/(time.perf_counter()-t0)
        cv2.putText(frame,f"{fps:4.1f} fps",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,WHITE,1)
        cv2.putText(frame,f"Conf: {last_conf:.2f}",(5,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Pos: ({last_fx},{last_fy})",(5,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,YELLW,1)

        if pending_second: overlay="Engaging secondary..."
        elif shot_time: overlay="Target Eliminated"
        elif countdown_start: overlay=str(max(1,COUNTDOWN_SEC-int(now_t-countdown_start)))
        else: overlay="Searching for target"
        cv2.putText(frame,overlay,(5,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,RED,2)

        # Draw boxes
        if primary_active and primary_bbox:
            x1,y1,x2,y2=primary_bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),BLUE,2)
            cv2.drawMarker(frame,primary_center,BLUE,cv2.MARKER_CROSS,12,2)
        if secondary_active and secondary_bbox:
            sx1,sy1,sx2,sy2=secondary_bbox
            cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),GREEN,2)
            cv2.drawMarker(frame,secondary_center,GREEN,cv2.MARKER_CROSS,12,2)
        elif pending_second and secondary_locked_center:
            cv2.drawMarker(frame,secondary_locked_center,GREEN,cv2.MARKER_CROSS,12,2)

        ret,jpeg=cv2.imencode(".jpg",frame)
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg.tobytes()+b"\r\n")

@app.route("/video_feed")
def video_feed():
    log("Client connected to /video_feed")
    return Response(gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

# ── MAIN ───────────────────────────────────────────────────────────────
if __name__=="__main__":
    try:
        threading.Thread(target=gen_frames, daemon=True).start()
        log("Tracking thread started (runs even without client)")
        log("Starting Flask stream on port 5000 ...")
        app.run(host="0.0.0.0",port=5000,threaded=True)
    finally:
        for ch in (PAN_CH,TILT_CH): pca.channels[ch].duty_cycle=0
        pca.deinit()
        GPIO.output(LED_PIN,GPIO.LOW); GPIO.cleanup()
        picam2.close()
        log("Clean shutdown")
