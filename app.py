import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("YOLOv8s Object Detection Online")

# رفع صورة
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تحميل الموديل (YOLOv8s)
    model = YOLO("yolov8s.pt")  # تأكدي ان الموديل موجود في المشروع أو على نفس المسار

    # الكشف
    results = model(image_np)

    # رسم النتائج على الصورة
    annotated_frame = results[0].plot()  # YOLOv8 outputs a list, نرسم على أول نتيجة

    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)