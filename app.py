import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.title("YOLOv8s Object Detection Online (No OpenCV)")

# تحميل الموديل مرة واحدة
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# رفع صورة أو فيديو
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # قراءة الصورة
    img = Image.open(uploaded_file).convert("RGB")
    
    # تحويل الصورة لمصفوفة NumPy لأن YOLO بيشتغل عليها
    img_np = np.array(img)
    
    # كشف الأجسام
    results = model(img_np)
    
    # عمل annotate للصورة
    annotated_img = results[0].plot()  # ترجع NumPy array
    
    # تحويلها مرة تانية لـ PIL Image للعرض
    annotated_img_pil = Image.fromarray(annotated_img)
    
    st.image(annotated_img_pil, caption="Detected Objects", use_column_width=True)