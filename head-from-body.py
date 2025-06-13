#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
import cv2
from picamera2 import Picamera2

# —– Load Haar upper‐body cascade ——————————————————————
body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
if body_cascade.empty():
    raise RuntimeError("Could not load haarcascade_upperbody.xml")

# —– Initialize PiCamera2 once ——————————————————————
picam2 = Picamera2()
cfg = picam2.preview_configuration
cfg.main.size   = (640, 480)
cfg.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# —– Flask setup ——————————————————————————————
app = Flask(__name__)
HTML = """
<html><head><title>Head from Body</title></head>
<body style="margin:0; background:#000;">
  <img src="{{ url_for('video_feed') }}"
       style="width:100vw; height:100vh; object-fit:contain;"/>
</body></html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

def gen_frames():
    while True:
        # 1) Grab frame from PiCamera2
        frame = picam2.capture_array()

        # 2) Convert and detect bodies
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # 3) Draw body + estimate head
        for (x, y, w, h) in bodies:
            # Full‐body box (blue)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (255, 0, 0), 2)

            # Head estimate (top 25% of body, centered 50% width)
            head_h = int(h * 0.25)
            head_w = int(w * 0.5)
            head_x = x + (w - head_w) // 2
            head_y = y

            # Head box (red)
            cv2.rectangle(frame, (head_x, head_y),
                          (head_x + head_w, head_y + head_h),
                          (0, 0, 255), 2)

        # 4) JPEG‐encode & yield for MJPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
