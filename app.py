#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
import cv2
from picamera2 import Picamera2

# —– Configuration ————————————————————————————
CAM_WIDTH, CAM_HEIGHT = 1920, 1080

# Load your Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
if face_cascade.empty() or body_cascade.empty():
    raise RuntimeError("Could not load one or more Haar cascades")

# Initialize PiCamera2
picam2 = Picamera2()
cfg = picam2.preview_configuration
cfg.main.size = (CAM_WIDTH, CAM_HEIGHT)
cfg.main.format = 'RGB888'
picam2.configure("preview")
picam2.start()

# —– Flask App ——————————————————————————————
app = Flask(__name__)

# Simple HTML page embedding the MJPEG stream
HTML = """
<html><head><title>Pi Stream</title></head>
<body style="margin:0; background:#000;">
  <img src="{{ url_for('video_feed') }}" style="width:100vw; height:100vh; object-fit:contain;"/>
</body></html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

def gen_frames():
    """Generator that yields MJPEG-compliant byte frames."""
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Face detection (red)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces)>0:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), 2)
        else:
            # fallback: upper-body detection in blue
            bodies = body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 50)
            )
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                            (255, 0, 0), 2)

        # JPEG-encode frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listen on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
