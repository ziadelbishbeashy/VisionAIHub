from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ml"
    / "traffic_density_model.pkl"
)

FEATURES_PATH = (
    BASE_DIR
    / "models"
    / "ml"
    / "traffic_features.pkl"
)


traffic_model = joblib.load(MODEL_PATH)
TRAFFIC_FEATURES = joblib.load(FEATURES_PATH)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def create_traffic_features(summary):
    persons = float(summary.get("persons", 0))
    cars = float(summary.get("cars", 0))
    buses = float(summary.get("buses", 0))
    trucks = float(summary.get("trucks", 0))
    motorcycles = float(summary.get("motorcycles", 0))
    bicycles = float(summary.get("bicycles", 0))
    traffic_lights = float(summary.get("traffic_lights", 0))
    stop_signs = float(summary.get("stop_signs", 0))

    total_vehicles = float(summary.get("total_vehicles", 0))
    total_people = float(summary.get("total_people", persons))
    total_objects = float(summary.get("total_objects", 0))

    image_width = float(summary.get("image_width", 0))
    image_height = float(summary.get("image_height", 0))

    image_area = image_width * image_height

    if image_area <= 0:
        image_area = 1.0

    vehicle_occupancy = float(
        summary.get("vehicle_occupancy", 0)
    )

    heavy_vehicle_ratio = safe_divide(
        buses + trucks,
        total_vehicles
    )

    person_vehicle_ratio = safe_divide(
        total_people,
        total_vehicles
    )

    avg_confidence = float(summary.get("avg_confidence", 0))
    min_confidence = float(summary.get("min_confidence", 0))
    max_confidence = float(summary.get("max_confidence", 0))

    # Engineered features used during training
    crowd_factor = persons * total_vehicles

    road_usage = (
        vehicle_occupancy
        * total_vehicles
    )

    vehicle_density = (
        total_vehicles
        / image_area
    )

    person_density = (
        persons
        / image_area
    )

    heavy_vehicle_count = buses + trucks

    vehicle_person_ratio = (
        total_vehicles
        / (persons + 1)
    )

    traffic_complexity = (
        total_objects
        * vehicle_occupancy
    )

    non_car_vehicles = (
        buses
        + trucks
        + motorcycles
        + bicycles
    )

    feature_values = {
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

        "vehicle_occupancy": vehicle_occupancy,
        "heavy_vehicle_ratio": heavy_vehicle_ratio,
        "person_vehicle_ratio": person_vehicle_ratio,

        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,

        "crowd_factor": crowd_factor,
        "road_usage": road_usage,
        "vehicle_density": vehicle_density,
        "person_density": person_density,
        "heavy_vehicle_count": heavy_vehicle_count,
        "vehicle_person_ratio": vehicle_person_ratio,
        "traffic_complexity": traffic_complexity,
        "non_car_vehicles": non_car_vehicles
    }

    missing_features = [
        feature
        for feature in TRAFFIC_FEATURES
        if feature not in feature_values
    ]

    if missing_features:
        raise ValueError(
            f"Missing traffic features: {missing_features}"
        )

    ordered_values = {
        feature: feature_values[feature]
        for feature in TRAFFIC_FEATURES
    }

    return pd.DataFrame([ordered_values])


def predict_traffic_density(summary):
    input_data = create_traffic_features(summary)

    predicted_label = str(
        traffic_model.predict(input_data)[0]
    )

    confidence = None
    class_probabilities = {}

    if hasattr(traffic_model, "predict_proba"):
        probabilities = traffic_model.predict_proba(
            input_data
        )[0]

        classes = [
            str(label)
            for label in traffic_model.classes_
        ]

        class_probabilities = {
            label: round(float(probability) * 100, 2)
            for label, probability
            in zip(classes, probabilities)
        }

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )

    return {
        "ml_traffic_density": predicted_label,
        "ml_traffic_confidence": confidence,
        "ml_class_probabilities": class_probabilities
    }