import cv2
import numpy as np
import os
import sys
import shutil

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)  # Ensure it's 0/1 or 0/255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w - 1, y + h - 1))

    return bboxes

def xyxy_to_yolo(x_min, y_min, x_max, y_max, img_w = 800, img_h = 800):
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def load_and_resize(path, label, size=(200,200)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    cv2.putText(img, label, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

def img_to_yolo_bbox(img_path, combined=False):
    org_img_path = img_path.replace('_mask', '').replace('ground_truth', 'test')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 800))
    if img is None:
        raise ValueError(f"Image at {img_path} could not be read.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_array = (mask > 0).astype(np.uint8)
    bboxes = mask_to_bbox(mask_array)   
    yolo_bboxes = []

    
    if combined:
        color = load_and_resize(r"mvtec_anomaly_detection\pill\test\color\000.png", "0")
        contamination = load_and_resize(r"mvtec_anomaly_detection\pill\test\contamination\000.png", "2")
        crack = load_and_resize(r"mvtec_anomaly_detection\pill\test\crack\000.png", "3")
        faulty_imprint = load_and_resize(r"mvtec_anomaly_detection\pill\test\faulty_imprint\000.png", "4")
        _pill_type = load_and_resize(r"mvtec_anomaly_detection\pill\test\pill_type\000.png", "5")
        scratch = load_and_resize(r"mvtec_anomaly_detection\pill\test\scratch\000.png", "6")

        # Show all at once in different windows
        cv2.imshow("0", color)
        cv2.imshow("2", contamination)
        cv2.imshow("3", crack)
        cv2.imshow("4", faulty_imprint)
        cv2.imshow("5", _pill_type)
        cv2.imshow("6", scratch)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            original_img = cv2.imread(org_img_path)
            cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Bounding Box", original_img)
            cv2.waitKey(1)
            pill_type = int(input("Enter pill type:"))
            yolo_bbox = xyxy_to_yolo(x_min, y_min, x_max, y_max, img.shape[1], img.shape[0])
            yolo_bboxes.append((pill_type, yolo_bbox))
        return yolo_bboxes
    else:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            yolo_bbox = xyxy_to_yolo(x_min, y_min, x_max, y_max, img.shape[1], img.shape[0])
            yolo_bboxes.append(yolo_bbox)
        
        return yolo_bboxes


dataset_path = "mvtec_anomaly_detection/pill"
new_dataset_path = "dataset"
good_img = []
i = {"color": 0,
     "good": 0,
     "combined": 0,
     "contamination": 0,
     "crack": 0,
     "faulty_imprint": 0,
     "pill_type": 0,
     "scratch": 0,}
type_dict = {"color": 0,
        "combined": 1,
        "contamination": 2,
        "crack": 3,
        "faulty_imprint": 4,
        "pill_type": 5,
        "scratch": 6,}
for f in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, f)):
        for pill_type in os.listdir(os.path.join(dataset_path, f)):
            for img in os.listdir(os.path.join(dataset_path, f, pill_type)):
                if pill_type == "good":
                    good_img.append(os.path.join(dataset_path, f, pill_type, img))
                    print(f"Good image: {os.path.join(dataset_path, f, pill_type, img)}")
                    shutil.copy(os.path.join(dataset_path, f, pill_type, img), os.path.join(new_dataset_path, "images", f"{pill_type}_{i[pill_type]}.png"))
                    with open(os.path.join(new_dataset_path, "labels", f"{pill_type}_{i[pill_type]}.txt"), 'w') as t:
                        pass
                elif f == "ground_truth":
                    mask_path = os.path.join(dataset_path, f, pill_type, img)
                    img_path = os.path.join(dataset_path, 'test', pill_type, img.replace('_mask', ''))
                    if pill_type == "combined":
                        yolo_bboxes = img_to_yolo_bbox(mask_path, combined=True)
                        shutil.copy(img_path, os.path.join(new_dataset_path, "images", f"{pill_type}_{i[pill_type]}.png"))
                        with open(os.path.join(new_dataset_path, "labels", f"{pill_type}_{i[pill_type]}.txt"), 'w') as t:
                            for num_type, bbox in yolo_bboxes:
                                t.write(f"{num_type} {' '.join(map(str, bbox))}\n")
                        print(f"GT: {mask_path}, YOLO BBoxes: {yolo_bboxes}")
                        print(f"Defected: {img_path}")
                    else:
                        yolo_bboxes = img_to_yolo_bbox(mask_path)
                        shutil.copy(img_path, os.path.join(new_dataset_path, "images", f"{pill_type}_{i[pill_type]}.png"))
                        with open(os.path.join(new_dataset_path, "labels", f"{pill_type}_{i[pill_type]}.txt"), 'w') as t:
                            for bbox in yolo_bboxes:
                                num_type  = type_dict[pill_type]
                                t.write(f"{num_type} {' '.join(map(str, bbox))}\n")
                        print(f"GT: {mask_path}, YOLO BBoxes: {yolo_bboxes}")
                        print(f"Defected: {img_path}")
                i[pill_type] += 1


# img = cv2.imread(r"mvtec_anomaly_detection\pill\ground_truth\color\016_mask.png")
# print(img.shape[1], img.shape[0])  # Print image width and height
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# mask_array = (mask > 0).astype(np.uint8)
# print(mask_array.shape)  # Print mask shape
# bboxes = mask_to_bbox(mask_array)   
# for bbox in bboxes:
#     x_min, y_min, x_max, y_max = bbox
#     yolo_bbox = xyxy_to_yolo(x_min, y_min, x_max, y_max)
#     print(f"YOLO bbox: {yolo_bbox}")

# print(img_to_yolo_bbox(r"mvtec_anomaly_detection\pill\ground_truth\color\016_mask.png"))