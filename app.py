from datetime import datetime
from uuid import uuid4
import os

import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session
)

from werkzeug.utils import secure_filename

from utils.detection import detect_objects
from utils.gemini_assistant import answer_with_gemini
from utils.ml_predictor import predict_traffic_density
from utils.scene_classifier import predict_scene
from utils.weather_classifier import predict_weather

from utils.analytics_manager import (
    append_analysis_record,
    build_analytics_summary
)


app = Flask(__name__)


app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "vision-ai-hub-development-key"
)


UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "jfif",
    "webp"
}


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def make_json_safe(value):
    """
    Convert NumPy and other non-JSON values
    into normal Python values for Flask sessions.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        return value.item()

    return value


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():
    return render_template(
        "about.html"
    )


# =========================================================
# ANALYTICS PAGE
# =========================================================

@app.route("/analytics")
def analytics():
    analytics_summary = (
        build_analytics_summary()
    )

    return render_template(
        "analytics.html",
        analytics=analytics_summary
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Remove the previous Gemini context.
    session.pop(
        "assistant_context",
        None
    )

    # -----------------------------------------------------
    # Validate uploaded image
    # -----------------------------------------------------

    if "image" not in request.files:
        return (
            "No image was uploaded.",
            400
        )

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return (
            "Please choose an image.",
            400
        )

    if not allowed_file(
        uploaded_file.filename
    ):
        return (
            "Only JPG, JPEG, PNG, JFIF and WEBP "
            "images are supported.",
            400
        )

    # -----------------------------------------------------
    # Selected analysis modules
    # -----------------------------------------------------

    selected_modules = request.form.getlist(
        "analysis_types"
    )

    allowed_modules = {
        "detection",
        "traffic",
        "weather",
        "scene"
    }

    selected_modules = [
        module
        for module in selected_modules
        if module in allowed_modules
    ]

    if not selected_modules:
        return (
            "Please select at least one "
            "analysis module.",
            400
        )

    # -----------------------------------------------------
    # YOLO confidence threshold
    # -----------------------------------------------------

    try:
        confidence = float(
            request.form.get(
                "confidence",
                0.35
            )
        )

    except (ValueError, TypeError):
        confidence = 0.35

    confidence = min(
        max(confidence, 0.10),
        0.90
    )

    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    original_filename = secure_filename(
        uploaded_file.filename
    )

    if not original_filename:
        return (
            "Invalid image filename.",
            400
        )

    unique_filename = (
        f"{uuid4().hex[:10]}_"
        f"{original_filename}"
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    uploaded_file.save(
        image_path
    )

    # -----------------------------------------------------
    # Determine which modules should run
    # -----------------------------------------------------

    run_traffic = (
        "traffic" in selected_modules
    )

    run_weather = (
        "weather" in selected_modules
    )

    run_scene = (
        "scene" in selected_modules
    )

    # Traffic Intelligence requires YOLO features.
    run_detection = (
        "detection" in selected_modules
        or run_traffic
    )

    # -----------------------------------------------------
    # Initial analysis summary
    # -----------------------------------------------------

    summary = {
        "analysis_id": datetime.now().strftime(
            "%Y%m%d%H%M%S"
        ),

        "selected_modules":
            selected_modules,

        "show_detection":
            run_detection,

        "traffic_selected":
            run_traffic,

        "weather_selected":
            run_weather,

        "scene_selected":
            run_scene,

        "original_image":
            image_path.replace(
                "\\",
                "/"
            ),

        # Default detection values.
        # These protect the templates and analytics
        # when YOLO was not selected.
        "persons": 0,
        "cars": 0,
        "buses": 0,
        "trucks": 0,
        "motorcycles": 0,
        "bicycles": 0,
        "traffic_lights": 0,
        "stop_signs": 0,
        "total_objects": 0,
        "total_vehicles": 0,
        "total_people": 0,
        "avg_confidence": 0,
        "min_confidence": 0,
        "max_confidence": 0,
        "vehicle_occupancy": 0,
        "vehicle_density": 0,
        "processing_time": 0,
        "most_detected_object": "None",

        # Default traffic values.
        "ml_traffic_density": None,
        "ml_traffic_confidence": None,
        "ml_class_probabilities": {},

        # Default weather values.
        "weather_label": None,
        "weather_confidence": None,
        "weather_probabilities": {},

        # Default scene values.
        "scene_label": None,
        "scene_confidence": None,
        "scene_probabilities": {}
    }

    result_path = None

    # -----------------------------------------------------
    # 1. YOLO object detection
    # -----------------------------------------------------

    if run_detection:
        try:
            (
                result_path,
                detection_summary
            ) = detect_objects(
                image_path,
                RESULT_FOLDER,
                confidence_threshold=confidence
            )

            summary.update(
                detection_summary
            )

            summary["show_detection"] = True

        except Exception as error:
            print(
                "YOLO detection error:",
                type(error).__name__,
                error
            )

            return (
                f"Object detection failed: {error}",
                500
            )

    # -----------------------------------------------------
    # 2. Random Forest traffic prediction
    # -----------------------------------------------------

    if run_traffic:
        try:
            traffic_result = (
                predict_traffic_density(
                    summary
                )
            )

            summary.update(
                traffic_result
            )

            print(
                "Traffic prediction:",
                traffic_result.get(
                    "ml_traffic_density"
                )
            )

            print(
                "Traffic confidence:",
                traffic_result.get(
                    "ml_traffic_confidence"
                )
            )

            print(
                "Traffic probabilities:",
                traffic_result.get(
                    "ml_class_probabilities"
                )
            )

        except Exception as error:
            print(
                "Traffic ML error:",
                type(error).__name__,
                error
            )

            summary.update({
                "ml_traffic_density":
                    "Unavailable",

                "ml_traffic_confidence":
                    None,

                "ml_class_probabilities":
                    {},

                "traffic_error":
                    str(error)
            })

    # -----------------------------------------------------
    # 3. Weather classification
    # -----------------------------------------------------

    if run_weather:
        try:
            weather_result = (
                predict_weather(
                    image_path
                )
            )

            summary.update(
                weather_result
            )

            print(
                "Weather prediction:",
                weather_result.get(
                    "weather_label"
                )
            )

            print(
                "Weather confidence:",
                weather_result.get(
                    "weather_confidence"
                )
            )

            print(
                "Weather probabilities:",
                weather_result.get(
                    "weather_probabilities"
                )
            )

        except Exception as error:
            print(
                "Weather classification error:",
                type(error).__name__,
                error
            )

            summary.update({
                "weather_label":
                    "Unavailable",

                "weather_confidence":
                    None,

                "weather_probabilities":
                    {},

                "weather_error":
                    str(error)
            })

    # -----------------------------------------------------
    # 4. Scene classification
    # -----------------------------------------------------

    if run_scene:
        try:
            scene_result = (
                predict_scene(
                    image_path
                )
            )

            summary.update(
                scene_result
            )

            print(
                "Scene prediction:",
                scene_result.get(
                    "scene_label"
                )
            )

            print(
                "Scene confidence:",
                scene_result.get(
                    "scene_confidence"
                )
            )

            print(
                "Scene probabilities:",
                scene_result.get(
                    "scene_probabilities"
                )
            )

        except Exception as error:
            print(
                "Scene classification error:",
                type(error).__name__,
                error
            )

            summary.update({
                "scene_label":
                    "Unavailable",

                "scene_confidence":
                    None,

                "scene_probabilities":
                    {},

                "scene_error":
                    str(error)
            })

    # -----------------------------------------------------
    # Normalize YOLO result image path
    # -----------------------------------------------------

    normalized_result_path = None

    if result_path:
        normalized_result_path = (
            str(result_path)
            .replace(
                "\\",
                "/"
            )
        )

    # -----------------------------------------------------
    # Determine available Gemini context
    # -----------------------------------------------------

    object_detection_available = (
        run_detection
        and "total_objects" in summary
    )

    traffic_available = (
        run_traffic
        and summary.get(
            "ml_traffic_density"
        ) not in {
            None,
            "Unavailable"
        }
    )

    weather_available = (
        run_weather
        and summary.get(
            "weather_label"
        ) not in {
            None,
            "Unavailable"
        }
    )

    scene_available = (
        run_scene
        and summary.get(
            "scene_label"
        ) not in {
            None,
            "Unavailable"
        }
    )

    # -----------------------------------------------------
    # Build Gemini structured context
    # -----------------------------------------------------

    assistant_context = {

        "analysis_information": {

            "analysis_id":
                summary.get(
                    "analysis_id"
                ),

            "selected_modules":
                selected_modules,

            "yolo_confidence_threshold":
                confidence
        },


        "object_detection": {

            "available":
                object_detection_available,

            "persons":
                summary.get(
                    "persons",
                    0
                ),

            "cars":
                summary.get(
                    "cars",
                    0
                ),

            "buses":
                summary.get(
                    "buses",
                    0
                ),

            "trucks":
                summary.get(
                    "trucks",
                    0
                ),

            "motorcycles":
                summary.get(
                    "motorcycles",
                    0
                ),

            "bicycles":
                summary.get(
                    "bicycles",
                    0
                ),

            "traffic_lights":
                summary.get(
                    "traffic_lights",
                    0
                ),

            "stop_signs":
                summary.get(
                    "stop_signs",
                    0
                ),

            "total_objects":
                summary.get(
                    "total_objects",
                    0
                ),

            "total_people":
                summary.get(
                    "total_people",
                    0
                ),

            "total_vehicles":
                summary.get(
                    "total_vehicles",
                    0
                ),

            "average_detection_confidence":
                summary.get(
                    "avg_confidence"
                ),

            "minimum_detection_confidence":
                summary.get(
                    "min_confidence"
                ),

            "maximum_detection_confidence":
                summary.get(
                    "max_confidence"
                ),

            "vehicle_occupancy":
                summary.get(
                    "vehicle_occupancy",
                    0
                ),

            "vehicle_share_percent":
                summary.get(
                    "vehicle_density",
                    0
                ),

            "most_detected_object":
                summary.get(
                    "most_detected_object"
                ),

            "processing_time_seconds":
                summary.get(
                    "processing_time",
                    0
                )
        },


        "traffic_analysis": {

            "available":
                traffic_available,

            "prediction":
                summary.get(
                    "ml_traffic_density"
                ),

            "confidence_percent":
                summary.get(
                    "ml_traffic_confidence"
                ),

            "class_probabilities":
                summary.get(
                    "ml_class_probabilities",
                    {}
                ),

            "error":
                summary.get(
                    "traffic_error"
                )
        },


        "weather_analysis": {

            "available":
                weather_available,

            "prediction":
                summary.get(
                    "weather_label"
                ),

            "confidence_percent":
                summary.get(
                    "weather_confidence"
                ),

            "class_probabilities":
                summary.get(
                    "weather_probabilities",
                    {}
                ),

            "error":
                summary.get(
                    "weather_error"
                )
        },


        "scene_analysis": {

            "available":
                scene_available,

            "prediction":
                summary.get(
                    "scene_label"
                ),

            "confidence_percent":
                summary.get(
                    "scene_confidence"
                ),

            "class_probabilities":
                summary.get(
                    "scene_probabilities",
                    {}
                ),

            "model_name":
                summary.get(
                    "scene_model_name"
                ),

            "error":
                summary.get(
                    "scene_error"
                )
        }
    }

    # -----------------------------------------------------
    # Store Gemini context in Flask session
    # -----------------------------------------------------

    session["assistant_context"] = (
        make_json_safe(
            assistant_context
        )
    )

    # -----------------------------------------------------
    # Save analysis for analytics dashboard
    # -----------------------------------------------------

    try:
        append_analysis_record(
            summary=summary,
            selected_modules=selected_modules
        )

    except Exception as error:
        print(
            "Analytics history error:",
            type(error).__name__,
            error
        )

    # -----------------------------------------------------
    # Model confidence values for result charts
    # -----------------------------------------------------

    model_confidences = {}

    if (
        run_traffic
        and summary.get(
            "ml_traffic_confidence"
        ) is not None
    ):
        model_confidences["Traffic"] = (
            summary[
                "ml_traffic_confidence"
            ]
        )

    if (
        run_weather
        and summary.get(
            "weather_confidence"
        ) is not None
    ):
        model_confidences["Weather"] = (
            summary[
                "weather_confidence"
            ]
        )

    if (
        run_scene
        and summary.get(
            "scene_confidence"
        ) is not None
    ):
        model_confidences["Scene"] = (
            summary[
                "scene_confidence"
            ]
        )

    # -----------------------------------------------------
    # Show results page
    # -----------------------------------------------------

    return render_template(
        "results.html",

        original_image=(
            image_path.replace(
                "\\",
                "/"
            )
        ),

        result_image=
            normalized_result_path,

        summary=
            summary,

        confidence=
            confidence,

        model_confidences=
            model_confidences
    )


# =========================================================
# GEMINI ASSISTANT
# =========================================================

@app.route(
    "/ask-assistant",
    methods=["POST"]
)
def ask_assistant():
    try:
        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        if not question:
            return jsonify({
                "success": False,
                "answer": (
                    "Please enter a question."
                )
            }), 400

        assistant_context = session.get(
            "assistant_context"
        )

        if not assistant_context:
            return jsonify({
                "success": False,
                "answer": (
                    "Please upload and analyze "
                    "an image before asking "
                    "the assistant."
                )
            }), 400

        answer = answer_with_gemini(
            question=question,
            context=assistant_context
        )

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "assistant_type": "Gemini"
        })

    except Exception as error:
        print(
            "Gemini assistant error:",
            type(error).__name__,
            error
        )

        error_text = str(
            error
        ).lower()

        temporary_error = any(
            marker in error_text
            for marker in [
                "503",
                "504",
                "unavailable",
                "high demand",
                "overloaded",
                "temporarily"
            ]
        )

        if temporary_error:
            return jsonify({
                "success": False,

                "answer": (
                    "Gemini is currently experiencing "
                    "high demand. Please wait a moment "
                    "and try your question again."
                ),

                "assistant_type":
                    "Temporarily unavailable"

            }), 503

        return jsonify({
            "success": False,

            "answer": (
                "The Gemini assistant could not answer. "
                "Check the Flask terminal for the error."
            ),

            "assistant_type":
                "Unavailable"

        }), 500


# =========================================================
# OLD GEMINI COMPATIBILITY ROUTE
# =========================================================

@app.route(
    "/ask-weather",
    methods=["POST"]
)
def ask_weather_compatibility():
    return ask_assistant()


# =========================================================
# HISTORY PAGE
# =========================================================

@app.route("/history")
def history():

    csv_path = (
        "data/object_counts.csv"
    )

    if os.path.exists(
        csv_path
    ):
        try:
            dataframe = pd.read_csv(
                csv_path
            )

            records = (
                dataframe
                .tail(20)
                .iloc[::-1]
                .to_dict(
                    orient="records"
                )
            )

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError
        ):
            records = []

    else:
        records = []

    return render_template(
        "history.html",
        records=records
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )