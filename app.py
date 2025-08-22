import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO

# Load YOLOv8
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Live Detection")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_small = cv2.resize(img, (640, 360))  # faster processing
        results = model(img_small, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = model.names[int(box.cls[0])]
                cv2.rectangle(img_small, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img_small, cls, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)