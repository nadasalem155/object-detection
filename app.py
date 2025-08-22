import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("YOLOv8s Object Detection Online")

# تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # خلي الملف في نفس الفولدر

model = load_model()

# رفع صورة أو فيديو
uploaded_file = st.file_uploader("Upload your video or image", type=["mp4", "mov", "avi", "jpg", "png"])

stframe = st.empty()

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if uploaded_file.type.startswith("image"):
        # لو الصورة
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(img)
        annotated_img = results[0].plot()
        stframe.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB")
    
    elif uploaded_file.type.startswith("video"):
        # لو الفيديو
        cap = cv2.VideoCapture(uploaded_file.name)  # أونلاين ممكن نحتاج tempfile هنا أحيانًا
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()