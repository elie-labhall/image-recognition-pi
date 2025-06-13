#!/usr/bin/env python3
import cv2
from picamera2 import Picamera2

# Define the 21 classes from PASCAL VOC
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Assign unique colors to each class for visualization
import random
COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in CLASSES]

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"
)

# Setup PiCamera2
picam2 = Picamera2()
cfg = picam2.preview_configuration
cfg.main.size = (320, 240)
cfg.main.format = 'RGB888'
picam2.configure("preview")
picam2.start()

print("🎯 Object detection started... press 'q' to quit.")

try:
    while True:
        frame = picam2.capture_array()

        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id >= len(CLASSES):
                    continue

                box = detections[0, 0, i, 3:7] * \
                    [frame.shape[1], frame.shape[0],
                     frame.shape[1], frame.shape[0]]
                (x1, y1, x2, y2) = box.astype("int")

                label = f"{CLASSES[class_id]}: {confidence:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(frame, label, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame = cv2.resize(frame, (1280, 720))  # upscale for display
        cv2.imshow("MobileNetSSD Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    picam2.close()
