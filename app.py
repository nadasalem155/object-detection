import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO

# Load YOLOv8 small model (higher accuracy than nano)
model = YOLO("yolov8s.pt")  # Use device="cuda:0" if GPU available

st.title("YOLOv8s Live Detection (Web)")

# WebRTC configuration for live streaming
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Frame counter to skip frames for speed
frame_count = 0

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        global frame_count
        # Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # Resize frame for faster processing
        img_small = cv2.resize(img, (640, 480))
        
        # Process every 2nd frame to reduce lag
        if frame_count % 2 == 0:
            results = model(img_small, conf=0.4, verbose=False)
            
            # Draw bounding boxes and labels
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = model.names[int(box.cls[0])]
                    conf = float(box.conf[0]) * 100
                    label = f"{cls} {conf:.1f}%"
                    cv2.rectangle(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_small, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_count += 1
        return cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

# Start live webcam streaming in Streamlit
webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)