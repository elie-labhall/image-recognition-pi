#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
import cv2
from picamera2 import Picamera2
import numpy as np

import time


CAM_WIDTH, CAM_HEIGHT = 1200, 900
CONF_THRESHOLD = 0.5
PERSON_CLASS_ID = 15

# Load DNN model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize camera
# --- Camera setup ---
picam2 = Picamera2()

video_cfg = picam2.create_video_configuration(
    main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
    buffer_count=4
)

picam2.configure(video_cfg)  # first configure with your config object

picam2.set_controls({"FrameRate": 60})  # then set controls separately

picam2.start()



# Flask setup
app = Flask(__name__)
HTML = """
<html><head><title>Pi Person Stream</title></head>
<body style="margin:0; background:#000;">
  <img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain;"/>
</body></html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

def gen_frames():
    while True:

        frame_start = time.perf_counter()  # Start timing the frame

        frame = picam2.capture_array()
        if frame is None:
            continue
 
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls  = int(detections[0, 0, i, 1])
            if conf > CONF_THRESHOLD and cls == PERSON_CLASS_ID:
                x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                break  # only first detection

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Measure frame processing time and calculate FPS
        frame_time = time.perf_counter() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        print(f"Resolution: {CAM_WIDTH}x{CAM_HEIGHT}, FPS: {fps:.2f}")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
