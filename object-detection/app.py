import streamlit as st
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Live Object Detection")

# Checkbox to start/stop webcam
run = st.checkbox("Start Webcam")
frame_placeholder = st.image([])

# Webcam loop only if checkbox is checked
if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("⚠ Cannot access webcam.")
            break

        # Run YOLO detection
        results = model(frame, conf=0.4)

        # Draw detections
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert BGR → RGB for Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()