import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Object Detection")

# Check if running locally or cloud
local_run = st.checkbox("Run Live Webcam (local only)")

if local_run:
    import cv2
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.image([])
    while local_run:
        ret, frame = cap.read()
        if not ret:
            st.write("⚠ Cannot access webcam.")
            break
        results = model(frame, conf=0.4)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    uploaded_file = st.camera_input("Take a picture (cloud compatible)")
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))
        results = model(img, conf=0.4)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # Draw rectangles on PIL image
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)