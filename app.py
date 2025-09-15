import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load model
model = YOLO("runs/detect/train5/weights/best.pt")

st.title("Defect Detection System (YOLO)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes,  1)
    
    results = model.predict(source=img, conf=0.25)

    st.image(results[0].plot(), channels="BGR")

    if len(results[0].boxes) == 0:
        st.warning("No defects detected.")
    else:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        st.subheader("Detection Details")
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            class_name = results[0].names[int(cls)]
            st.write(f"- **Class:** {class_name} | **Conf:** {conf:.2f} | **BBox:** ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")