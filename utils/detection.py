from ultralytics import YOLO
from collections import Counter
import cv2
import os
import csv
import time
from datetime import datetime

# Load YOLO model once
model = YOLO("yolov8n.pt")

# Smart City classes
TARGET_CLASSES = {
    "person": "persons",
    "car": "cars",
    "bus": "buses",
    "truck": "trucks",
    "motorcycle": "motorcycles",
    "bicycle": "bicycles",
    "traffic light": "traffic_lights",
    "stop sign": "stop_signs"
}

CSV_PATH = "data/object_counts.csv"


def initialize_csv():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "date",
                "filename",
                "original_image",
                "result_image",

                "persons",
                "cars",
                "buses",
                "trucks",
                "motorcycles",
                "bicycles",
                "traffic_lights",
                "stop_signs",

                "total_vehicles",
                "total_people",
                "total_objects",

                "avg_confidence",
                "min_confidence",
                "max_confidence",

                "vehicle_density",
                "most_detected_object",
                "processing_time",

                "traffic_level",
                "risk_level"
            ])


def get_existing_records_count():
    if not os.path.exists(CSV_PATH):
        return 0

    with open(CSV_PATH, "r", encoding="utf-8") as file:
        return max(0, sum(1 for _ in file) - 1)


def predict_traffic(total_vehicles):
    """
    Temporary rule-based prediction.
    Later this will be replaced by the trained ML model.
    """
    if total_vehicles >= 12:
        return "High"
    elif total_vehicles >= 5:
        return "Medium"
    else:
        return "Low"


def predict_risk(total_people, total_vehicles):
    """
    Temporary rule-based prediction.
    Later this will be replaced by the trained ML model.
    """
    if total_people >= 5 and total_vehicles >= 10:
        return "High"
    elif total_people >= 2 and total_vehicles >= 5:
        return "Medium"
    else:
        return "Low"


def detect_objects(image_path, output_folder, confidence_threshold=0.4):
    initialize_csv()

    start_time = time.time()

    results = model(image_path, conf=confidence_threshold)
    result = results[0]

    counts = Counter()
    confidences = []

    # Count detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name in TARGET_CLASSES:
            counts[TARGET_CLASSES[class_name]] += 1
            confidences.append(confidence)

    # Save detected image
    detected_image = result.plot()

    output_filename = "detected_" + os.path.basename(image_path)

    output_path = os.path.join(
        output_folder,
        output_filename
    )

    cv2.imwrite(output_path, detected_image)

    # Extract counts
    persons = counts["persons"]
    cars = counts["cars"]
    buses = counts["buses"]
    trucks = counts["trucks"]
    motorcycles = counts["motorcycles"]
    bicycles = counts["bicycles"]
    traffic_lights = counts["traffic_lights"]
    stop_signs = counts["stop_signs"]

    # Main statistics
    total_vehicles = cars + buses + trucks + motorcycles + bicycles
    total_people = persons
    total_objects = sum(counts.values())

    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
    min_confidence = round(min(confidences), 2) if confidences else 0
    max_confidence = round(max(confidences), 2) if confidences else 0

    vehicle_density = round((total_vehicles / total_objects) * 100, 2) if total_objects else 0

    if total_objects > 0:
        most_detected_key = max(counts, key=counts.get)
        most_detected_object = most_detected_key.replace("_", " ").title()
    else:
        most_detected_object = "None"

    processing_time = round(time.time() - start_time, 2)

    # Temporary prediction
    traffic_level = predict_traffic(total_vehicles)
    risk_level = predict_risk(total_people, total_vehicles)

    analysis_id = get_existing_records_count() + 1

    # Save row to CSV
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        os.path.basename(image_path),
        image_path.replace("\\", "/"),
        output_path.replace("\\", "/"),

        persons,
        cars,
        buses,
        trucks,
        motorcycles,
        bicycles,
        traffic_lights,
        stop_signs,

        total_vehicles,
        total_people,
        total_objects,

        avg_confidence,
        min_confidence,
        max_confidence,

        vehicle_density,
        most_detected_object,
        processing_time,

        traffic_level,
        risk_level
    ]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    summary = {
        "analysis_id": analysis_id,

        "persons": persons,
        "cars": cars,
        "buses": buses,
        "trucks": trucks,
        "motorcycles": motorcycles,
        "bicycles": bicycles,
        "traffic_lights": traffic_lights,
        "stop_signs": stop_signs,

        "total_vehicles": total_vehicles,
        "total_people": total_people,
        "total_objects": total_objects,

        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,

        "vehicle_density": vehicle_density,
        "most_detected_object": most_detected_object,
        "processing_time": processing_time,

        "traffic_level": traffic_level,
        "risk_level": risk_level,

        "original_image": image_path.replace("\\", "/"),
        "result_image": output_path.replace("\\", "/")
    }

    return output_path, summary