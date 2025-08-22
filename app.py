import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import cv2  # فقط لاستخدام دوال الرسم، مش VideoCapture
import numpy as np

st.set_page_config(page_title="YOLOv8s Live Detection", layout="wide")
st.title("YOLOv8s Live Detection (Web) with Box Smoothing")

# Load YOLOv8 small model
model = YOLO("yolov8s.pt")  # You can use device="cuda:0" if GPU is available

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

frame_count = 0

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_boxes = []
        self.prev_labels = []

    def smooth_boxes(self, prev, current, alpha=0.6):
        smoothed = []
        for i in range(len(current)):
            if i < len(prev):
                x1 = int(alpha * current[i][0] + (1 - alpha) * prev[i][0])
                y1 = int(alpha * current[i][1] + (1 - alpha) * prev[i][1])
                x2 = int(alpha * current[i][2] + (1 - alpha) * prev[i][2])
                y2 = int(alpha * current[i][3] + (1 - alpha) * prev[i][3])
                smoothed.append((x1, y1, x2, y2))
            else:
                smoothed.append(current[i])
        return smoothed

    def recv(self, frame):
        global frame_count
        img = frame.to_ndarray(format="bgr24")
        img_small = cv2.resize(img, (640, 480))

        boxes_current = []
        labels_current = []

        if frame_count % 2 == 0:
            results = model(img_small, conf=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = model.names[int(box.cls[0])]
                    conf = float(box.conf[0]) * 100
                    boxes_current.append((x1, y1, x2, y2))
                    labels_current.append(f"{cls} {conf:.1f}%")

            boxes_smooth = self.smooth_boxes(self.prev_boxes, boxes_current)
            self.prev_boxes = boxes_smooth
            self.prev_labels = labels_current
        else:
            boxes_smooth = self.prev_boxes
            labels_current = self.prev_labels

        for i, (x1, y1, x2, y2) in enumerate(boxes_smooth):
            cv2.rectangle(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_small, labels_current[i], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_count += 1
        return cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)


# Start streaming via WebRTC (browser camera)
webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)