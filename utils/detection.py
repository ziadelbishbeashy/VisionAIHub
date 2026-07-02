from builtins import len, round, str, sum
from collections import Counter
from datetime import datetime
from pathlib import Path

import csv
import os
import time

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "object_counts.csv"

YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"


# ---------------------------------------------------------
# Load YOLO once when Flask starts
# ---------------------------------------------------------

model = YOLO(str(YOLO_MODEL_PATH))


# ---------------------------------------------------------
# Smart-city classes
# ---------------------------------------------------------

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


# Only ask YOLO to detect the classes used by VisionAIHub.
TARGET_CLASS_IDS = [
    class_id
    for class_id, class_name in model.names.items()
    if class_name in TARGET_CLASSES
]


VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}


# ---------------------------------------------------------
# CSV setup
# ---------------------------------------------------------

CSV_COLUMNS = [
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
]


def initialize_csv():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        with CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)


def get_existing_records_count():
    if not CSV_PATH.exists():
        return 0

    with CSV_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        return max(
            0,
            sum(1 for _ in file) - 1
        )


# ---------------------------------------------------------
# Temporary rule-based values
# These remain only for old dashboard/history compatibility.
# The Random Forest result will be added separately in app.py.
# ---------------------------------------------------------

def predict_traffic(total_vehicles):
    if total_vehicles >= 12:
        return "High"

    if total_vehicles >= 5:
        return "Medium"

    return "Low"


def predict_risk(total_people, total_vehicles):
    if total_people >= 5 and total_vehicles >= 10:
        return "High"

    if total_people >= 2 and total_vehicles >= 5:
        return "Medium"

    return "Low"


# ---------------------------------------------------------
# Main YOLO function
# ---------------------------------------------------------

def detect_objects(
    image_path,
    output_folder,
    confidence_threshold=0.35
):
    initialize_csv()

    start_time = time.time()

    image_path = Path(image_path)
    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read image to obtain dimensions.
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read the uploaded image: {image_path}"
        )

    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height

    if image_area <= 0:
        raise ValueError(
            "Uploaded image has invalid dimensions."
        )

    # Run YOLO using the same confidence threshold used
    # while generating the traffic ML training CSV.
    results = model(
        str(image_path),
        conf=confidence_threshold,
        classes=TARGET_CLASS_IDS,
        verbose=False
    )

    result = results[0]

    counts = Counter()
    confidences = []

    # Sum of vehicle bounding-box areas.
    # This is needed to reproduce vehicle_occupancy
    # from the machine-learning training dataset.
    vehicle_box_area = 0.0

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name not in TARGET_CLASSES:
            continue

        csv_name = TARGET_CLASSES[class_name]

        counts[csv_name] += 1
        confidences.append(confidence)

        if class_name in VEHICLE_CLASSES:
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            box_width = max(
                0.0,
                float(x2 - x1)
            )

            box_height = max(
                0.0,
                float(y2 - y1)
            )

            vehicle_box_area += (
                box_width * box_height
            )

    # -----------------------------------------------------
    # Save annotated YOLO result
    # -----------------------------------------------------

    detected_image = result.plot()

    output_filename = (
        "detected_"
        + image_path.name
    )

    output_path = (
        output_folder
        / output_filename
    )

    saved = cv2.imwrite(
        str(output_path),
        detected_image
    )

    if not saved:
        raise IOError(
            f"Could not save detection image: {output_path}"
        )

    # -----------------------------------------------------
    # Individual object counts
    # -----------------------------------------------------

    persons = counts["persons"]
    cars = counts["cars"]
    buses = counts["buses"]
    trucks = counts["trucks"]
    motorcycles = counts["motorcycles"]
    bicycles = counts["bicycles"]
    traffic_lights = counts["traffic_lights"]
    stop_signs = counts["stop_signs"]

    # -----------------------------------------------------
    # Main statistics
    # -----------------------------------------------------

    total_vehicles = (
        cars
        + buses
        + trucks
        + motorcycles
        + bicycles
    )

    total_people = persons
    total_objects = sum(counts.values())

    if confidences:
        avg_confidence = round(
            sum(confidences) / len(confidences),
            4
        )

        min_confidence = round(
            min(confidences),
            4
        )

        max_confidence = round(
            max(confidences),
            4
        )
    else:
        avg_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0

    # Percentage of detected objects that are vehicles.
    # This remains for the existing dashboard.
    vehicle_density = round(
        (total_vehicles / total_objects) * 100,
        2
    ) if total_objects else 0.0

    # Exact feature used by the trained Random Forest:
    # total vehicle bounding-box area / complete image area.
    vehicle_occupancy = round(
        vehicle_box_area / image_area,
        4
    )

    heavy_vehicle_ratio = round(
        (buses + trucks) / total_vehicles,
        4
    ) if total_vehicles else 0.0

    person_vehicle_ratio = round(
        total_people / total_vehicles,
        4
    ) if total_vehicles else 0.0

    if total_objects > 0:
        most_detected_key = max(
            counts,
            key=counts.get
        )

        most_detected_object = (
            most_detected_key
            .replace("_", " ")
            .title()
        )
    else:
        most_detected_object = "None"

    processing_time = round(
        time.time() - start_time,
        2
    )

    # Temporary rule-based values kept until history is
    # migrated to the unified ML-ready format.
    traffic_level = predict_traffic(
        total_vehicles
    )

    risk_level = predict_risk(
        total_people,
        total_vehicles
    )

    analysis_id = (
        get_existing_records_count() + 1
    )

    normalized_image_path = (
        str(image_path)
        .replace("\\", "/")
    )

    normalized_output_path = (
        str(output_path)
        .replace("\\", "/")
    )

    # -----------------------------------------------------
    # Save current backward-compatible history row
    # -----------------------------------------------------

    row = [
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        image_path.name,
        normalized_image_path,
        normalized_output_path,

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

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # -----------------------------------------------------
    # Summary passed to Flask, scene model and ML model
    # -----------------------------------------------------

    summary = {
        "analysis_id": analysis_id,

        # Image information required by ML
        "image_width": image_width,
        "image_height": image_height,
        "image_area": image_area,

        # Raw YOLO object counts
        "persons": persons,
        "cars": cars,
        "buses": buses,
        "trucks": trucks,
        "motorcycles": motorcycles,
        "bicycles": bicycles,
        "traffic_lights": traffic_lights,
        "stop_signs": stop_signs,

        # Main totals
        "total_vehicles": total_vehicles,
        "total_people": total_people,
        "total_objects": total_objects,

        # Confidence values
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,

        # ML-ready calculated values
        "vehicle_occupancy": vehicle_occupancy,
        "heavy_vehicle_ratio": heavy_vehicle_ratio,
        "person_vehicle_ratio": person_vehicle_ratio,

        # Dashboard values
        "vehicle_density": vehicle_density,
        "most_detected_object": most_detected_object,
        "processing_time": processing_time,

        # Temporary rule values
        "traffic_level": traffic_level,
        "risk_level": risk_level,

        # Image paths
        "original_image": normalized_image_path,
        "result_image": normalized_output_path
    }

    return normalized_output_path, summary