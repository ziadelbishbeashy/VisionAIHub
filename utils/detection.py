from ultralytics import YOLO
import os
from collections import Counter

model = YOLO("yolov8n.pt")

TARGET_CLASSES = [
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "traffic light"
]

def detect_objects(image_path, output_folder):
    results = model(image_path, conf=0.4)
    result = results[0]

    counts = Counter()

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in TARGET_CLASSES:
            counts[class_name] += 1

    result_image = result.plot()

    output_name = "detected_" + os.path.basename(image_path)
    output_path = os.path.join(output_folder, output_name)

    import cv2
    cv2.imwrite(output_path, result_image)

    return output_path, dict(counts)