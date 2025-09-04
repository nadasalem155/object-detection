import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np

def main():
    st.title("YOLOv8 Live Object Detection")

    # Load YOLO model
    model = YOLO("yolov8s.pt")

    # Open camera
    cap = cv2.VideoCapture(0)

    # Create a placeholder for video frames
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Camera not found")
            break

        # Run YOLO detection
        results = model(frame, conf=0.4, verbose=False)

        # Draw bounding boxes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = model.names[int(box.cls[0])]
                conf = float(box.conf[0]) * 100
                label = f"{cls} {conf:.1f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert BGR → RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()