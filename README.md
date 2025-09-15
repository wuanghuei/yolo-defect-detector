# Defect Detection System with YOLO (Ongoing)

This is an ongoing project exploring defect detection in manufactured products using **YOLOv8**.  
The goal is to automatically detect defects in objects such as pills and screws...etc, and to build a two-stage pipeline:

1. **Object Classification** – Identify which type of product is present in the image (e.g., pill or screw).  
2. **Defect Detection** – Load the corresponding YOLO weights for that product and detect possible defects.  

---

## Current Progress
- ✅ Collected and prepared dataset (~400+ images per class).  
- ✅ Converted segmentation mask labels into YOLO-compatible bounding boxes.  
- ✅ Re-labeled “combination” class into separate classes → improved **mAP@50 from 69% → 80%**.  
- ✅ Built a basic **Streamlit Demo** for real-time inference.  
- ☐ Currently extending system with **classification stage** to route inputs to product-specific defect models.  
- ☐ Deploy with Docker + Streamlit

---

## Tech Stack
- [YOLOv8](https://github.com/ultralytics/ultralytics)  
- Python, OpenCV  
- Streamlit (web app)  
---

## Demo
**Detection Result Example**  
<img src="results\defected.png" width="400"/>  

If no defect is detected, the system outputs:  


<img src="results\good.png" width="400"/>  

