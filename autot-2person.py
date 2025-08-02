#!/usr/bin/env python3
"""
Pi dual-target tracker (face-first) with relay trigger

• Detects faces first (SSD-ResNet10). Any body box that fully contains
  a face box is **ignored** so a single person is not counted twice.
• Tracks *up to two* independent targets:
    – Primary  (highest-priority)  → red rectangle + red cross-hair
    – Secondary (if present)       → orange rectangle + orange cross-hair
• Countdown 3-2-1 → fire primary → auto-retarget secondary (0.2 s) → fire
• Full MJPEG stream at  http://<pi-ip>:5000/video_feed
"""

# ── CONFIG ──────────────────────────────────────────────────────────────
PAN_CH, TILT_CH          = 0, 1
PAN_MIN, PAN_MAX         = 20, 160
TILT_MIN, TILT_MAX       = 50, 130
PAN_CENTER, TILT_CENTER  = 90, 90
TILT_DIR                 = -1          # flip if tilt reversed

CENTER_TOL_X, CENTER_TOL_Y = 8, 8      # ±px dead-band
GAIN_PAN,  GAIN_TILT       = 0.10, 0.10
LOST_TIMEOUT               = 2.0       # seconds → scan

CAM_W, CAM_H               = 300, 225

# scanning pattern when idle
SCAN_RANGE_PAN , SCAN_RANGE_TILT = 40, 30
SCAN_STEP, SCAN_WAVES, SCAN_DELAY = 1, 4, 0.10
scan_dir = 1                           # 1→right, −1→left

# face detector (primary)
FACE_PROTO   = "deploy.prototxt"
FACE_MODEL   = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_CONF    = 0.5
MIN_FACE_PIX = 400
FACE_HOLD_SEC= 0.6                     # hysteresis

# body detector (fallback/secondary)
BODY_PROTO   = "MobileNetSSD_deploy.prototxt"
BODY_MODEL   = "MobileNetSSD_deploy.caffemodel"
BODY_CLASS   = 15
BODY_CONF    = 0.5
MIN_BODY_FRAC= 0.02                    # ≥2 % frame

# firing
RELAY_PIN           = 6
COUNTDOWN_SEC       = 3
LOSS_TOL_FRAMES     = 2                # frames tolerated during countdown
SHOT_DURATION       = 0.25
SECOND_DELAY        = 0.2              # delay before secondary shot
ELIM_DISPLAY_SEC    = 1.0

# ── IMPORTS ─────────────────────────────────────────────────────────────
import math, time, cv2, numpy as np, board, busio, RPi.GPIO as GPIO
from pathlib import Path
from picamera2 import Picamera2
from adafruit_pca9685 import PCA9685
from flask import Flask, Response, render_template_string

# ── SERVO SETUP ─────────────────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c); pca.frequency = 50
def set_angle(ch:int, deg:float):
    deg = max(0,min(180,deg))
    pca.channels[ch].duty_cycle = int((500 + (deg/180)*1900)*65535/20000)
pan, tilt = PAN_CENTER, TILT_CENTER
set_angle(PAN_CH, pan); set_angle(TILT_CH, tilt)

# ── CAMERA ──────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size":(CAM_W, CAM_H),"format":"RGB888"}, buffer_count=1))
picam2.start()
CX, CY = CAM_W//2, CAM_H//2

# ── LOAD DNNs ───────────────────────────────────────────────────────────
for f in (FACE_PROTO, FACE_MODEL, BODY_PROTO, BODY_MODEL):
    if not Path(f).exists(): raise FileNotFoundError(f"Missing {f}")

net_face = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
net_body = cv2.dnn.readNetFromCaffe(BODY_PROTO, BODY_MODEL)
for n in (net_face, net_body):
    n.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    n.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ── GPIO / RELAY ────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)
def fire():
    GPIO.output(RELAY_PIN, GPIO.HIGH); time.sleep(SHOT_DURATION)
    GPIO.output(RELAY_PIN, GPIO.LOW)

# ── FLASK UI ────────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = """<!doctype html><html><head><title>Dual Target</title></head>
<body style="margin:0;background:#000;">
<img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain;">
</body></html>"""
@app.route("/")            # landing page
def index(): return render_template_string(HTML)

# ── STATE VARS ──────────────────────────────────────────────────────────
last_seen         = 0.0
last_face_time    = 0.0
last_face_box     = None

countdown_start   = None
shot_time         = None
pending_second    = False
second_start_t    = 0.0

lost_frames       = 0
last_fx, last_fy  = CX, CY
last_conf         = 0.0

# ── GEOMETRY HELPERS ────────────────────────────────────────────────────
def encloses(big, small):
    """True if rectangle big completely contains rectangle small."""
    bx1,by1,bx2,by2 = big
    sx1,sy1,sx2,sy2 = small
    return bx1<=sx1 and by1<=sy1 and bx2>=sx2 and by2>=sy2

# ── DETECTION HELPERS ───────────────────────────────────────────────────
def detect_faces(frame):
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104,177,123))
    net_face.setInput(blob); det=net_face.forward()
    out=[]
    for i in range(det.shape[2]):
        conf=float(det[0,0,i,2])
        if conf<FACE_CONF:continue
        x1,y1,x2,y2=(det[0,0,i,3:7]*np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
        if (x2-x1)*(y2-y1)<MIN_FACE_PIX:continue
        out.append((conf,(x1,y1,x2,y2)))
    out.sort(reverse=True,key=lambda x:x[0])
    return out[:2]

def detect_bodies(frame):
    blob=cv2.dnn.blobFromImage(frame,0.007843,(300,300),127.5)
    net_body.setInput(blob); det=net_body.forward()
    out=[]
    for i in range(det.shape[2]):
        conf=float(det[0,0,i,2]); cls=int(det[0,0,i,1])
        if conf<BODY_CONF or cls!=BODY_CLASS:continue
        x1,y1,x2,y2=(det[0,0,i,3:7]*np.array([CAM_W,CAM_H,CAM_W,CAM_H])).astype(int)
        if (x2-x1)*(y2-y1)<MIN_BODY_FRAC*CAM_W*CAM_H:continue
        out.append((conf,(x1,y1,x2,y2)))
    out.sort(reverse=True,key=lambda x:x[0])
    return out[:2]

# ── STREAM GENERATOR ────────────────────────────────────────────────────
def gen_frames():
    global pan, tilt, scan_dir, last_seen, last_face_time, last_face_box
    global countdown_start, shot_time, pending_second, second_start_t
    global lost_frames, last_fx, last_fy, last_conf

    while True:
        t0=time.perf_counter()
        frame=picam2.capture_array()

        # 1) FACE DETECTION ------------------------------------------------
        targets=[]
        for conf,box in detect_faces(frame):
            targets.append((conf,box,"face"))
        if not targets and last_face_box and time.time()-last_face_time<FACE_HOLD_SEC:
            targets.append((0.0,last_face_box,"face"))

        # 2) BODY DETECTION (filter those enclosing a face) ---------------
        if len(targets)<2:
            bodies=detect_bodies(frame)
            for conf,box in bodies:
                # skip body if it encloses any face box
                if any(encloses(box, fbox) for _,fbox,_ in targets):
                    continue
                if len(targets)<2:
                    targets.append((conf,box,"body"))

        # sort by confidence, keep top 2
        targets.sort(reverse=True,key=lambda x:x[0])
        if len(targets)>2: targets=targets[:2]

        primary  = targets[0] if targets else None
        secondary= targets[1] if len(targets)>1 else None
        detected = primary is not None

        # --- TRACKING ---------------------------------------------------
        if detected:
            lost_frames=0
            conf,(x1,y1,x2,y2),_ = primary
            fx,fy=(x1+x2)//2,(y1+y2)//2
            last_face_time=time.time() if primary[2]=="face" else last_face_time
            last_face_box = (x1,y1,x2,y2) if primary[2]=="face" else last_face_box

            # draw boxes
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)  # red primary
            if secondary:
                sx1,sy1,sx2,sy2=secondary[1]
                cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(0,165,255),2) # orange

            # servo adjust
            dx,dy=fx-CX,fy-CY
            if abs(dx)>CENTER_TOL_X:
                pan-=dx*GAIN_PAN; pan=max(PAN_MIN,min(PAN_MAX,pan)); set_angle(PAN_CH,pan)
            if abs(dy)>CENTER_TOL_Y:
                tilt+=TILT_DIR*dy*GAIN_TILT; tilt=max(TILT_MIN,min(TILT_MAX,tilt)); set_angle(TILT_CH,tilt)

            last_seen=time.time(); last_conf=conf; last_fx,last_fy=fx,fy

            # --- SHOT FSM ----------------------------------------------
            if pending_second:
                if time.time()-second_start_t>=SECOND_DELAY:
                    fire(); pending_second=False; shot_time=time.time()
            else:
                if shot_time:
                    if time.time()-shot_time>=ELIM_DISPLAY_SEC:
                        shot_time=None; countdown_start=time.time()
                elif countdown_start is None:
                    countdown_start=time.time()
                elif time.time()-countdown_start>=COUNTDOWN_SEC:
                    fire(); shot_time=time.time(); countdown_start=None
                    if secondary:
                        pending_second=True; second_start_t=time.time()
                        # quick aim to secondary
                        sx,sy=((sx1+sx2)//2,(sy1+sy2)//2)
                        pan-= (sx-CX)*GAIN_PAN
                        tilt+= TILT_DIR*(sy-CY)*GAIN_TILT
                        pan=max(PAN_MIN,min(PAN_MAX,pan)); set_angle(PAN_CH,pan)
                        tilt=max(TILT_MIN,min(TILT_MAX,tilt)); set_angle(TILT_CH,tilt)

        else:
            lost_frames+=1
            if lost_frames>LOSS_TOL_FRAMES:
                countdown_start=None; shot_time=None; pending_second=False
            if time.time()-last_seen>LOST_TIMEOUT:
                pan+=SCAN_STEP*scan_dir
                if pan>PAN_MAX or pan<PAN_MIN:
                    scan_dir*=-1; pan+=SCAN_STEP*scan_dir
                t_norm=(pan-PAN_CENTER)/SCAN_RANGE_PAN
                tilt=TILT_CENTER+SCAN_RANGE_TILT*math.sin(math.pi*SCAN_WAVES*t_norm)
                tilt=max(TILT_MIN,min(TILT_MAX,tilt))
                set_angle(PAN_CH,pan); set_angle(TILT_CH,tilt)
                time.sleep(SCAN_DELAY)

        # --- OVERLAYS --------------------------------------------------
        fps=1.0/(time.perf_counter()-t0)
        cv2.putText(frame,f"{fps:4.1f} fps",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,f"Conf: {last_conf:.2f}",(5,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Pos: ({last_fx},{last_fy})",(5,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        if pending_second:
            overlay="Engaging secondary..."
        elif shot_time:
            overlay="Target Eliminated"
        elif countdown_start:
            overlay=str(max(1,COUNTDOWN_SEC-int(time.time()-countdown_start)))
        else:
            overlay="Searching for target"
        cv2.putText(frame,overlay,(5,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # cross-hairs
        def draw_cross(x,y,color):
            sz,th=20,2
            cv2.line(frame,(x-sz,y),(x-5,y),color,th)
            cv2.line(frame,(x+5,y),(x+sz,y),color,th)
            cv2.line(frame,(x,y-sz),(x,y-5),color,th)
            cv2.line(frame,(x,y+5),(x,y+sz),color,th)
            cv2.circle(frame,(x,y),8,color,1)

        draw_cross(last_fx,last_fy,(0,0,255))           # red primary cross
        if secondary:
            sx,sy=((sx1+sx2)//2,(sy1+sy2)//2)
            draw_cross(sx,sy,(0,165,255))               # orange secondary

        # stream
        ret,jpeg=cv2.imencode(".jpg",frame)
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg.tobytes()+b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

# ── MAIN ────────────────────────────────────────────────────────────────
if __name__=="__main__":
    try:
        app.run(host="0.0.0.0",port=5000,threaded=True)
    finally:
        for ch in (PAN_CH,TILT_CH): pca.channels[ch].duty_cycle=0
        pca.deinit()
        GPIO.output(RELAY_PIN,GPIO.LOW); GPIO.cleanup()
        picam2.close()
        print("Clean shutdown")
