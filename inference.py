from ultralytics import YOLO
import cv2

# Load trained weights
model = YOLO("runs/detect/train5/weights/best.pt")

def predict(image_path):
    results = model.predict(source=image_path, conf=0.25)
    return results
