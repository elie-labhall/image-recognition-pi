import cv2
import time
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO core
core = Core()
model_path = "models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
model = core.read_model(model=model_path)

# ✅ Fix: Disable weight compression to avoid Unknown socket id
compiled_model = core.compile_model(model=model, device_name="CPU", config={"CPU_DISABLE_WEIGHT_COMPRESSION": "YES"})

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
frame_count = 0

print("🎥 Running face detection. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to model input shape
        input_image = cv2.resize(frame, (672, 384))
        input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

        # Inference
        result = compiled_model([input_image])[output_layer]
        detections = result[0][0]

        # Draw detections
        for detection in detections:
            confidence = float(detection[2])
            if confidence > 0.5:
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show FPS
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            print(f"📈 FPS: {fps:.2f}")
            prev_time = curr_time
            frame_count = 0

        # Display
        cv2.imshow("OpenVINO Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
