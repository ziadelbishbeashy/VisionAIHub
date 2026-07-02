import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "weather_model.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "weather_class_names.json"
)


# ---------------------------------------------------------
# Check required files
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Weather model was not found: {MODEL_PATH}"
    )


if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        "Weather class-names file was not found: "
        f"{CLASS_NAMES_PATH}"
    )


# ---------------------------------------------------------
# Load model and class names once
# ---------------------------------------------------------

weather_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    CLASS_NAMES = json.load(file)


if not isinstance(CLASS_NAMES, list):
    raise ValueError(
        "weather_class_names.json must contain a list."
    )


if len(CLASS_NAMES) != weather_model.output_shape[-1]:
    raise ValueError(
        "The number of class names does not match "
        "the weather model output size."
    )


print(
    "Weather model loaded successfully."
)

print(
    "Weather classes:",
    CLASS_NAMES
)


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_weather(image_path):
    """
    Predict the weather condition of one image.

    Returns:
        {
            "weather_label": str,
            "weather_confidence": float,
            "weather_probabilities": dict
        }
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Weather input image was not found: {image_path}"
        )


    # Load the image using the same size used during training.
    loaded_image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb"
    )


    image_array = tf.keras.utils.img_to_array(
        loaded_image
    )


    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Do not apply preprocess_input or divide by 255.
    # The new EfficientNetB0 model handles rescaling internally.
    image_array = image_array.astype(
        np.float32
    )


    predictions = weather_model.predict(
        image_array,
        verbose=0
    )[0]


    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "The weather model output does not match "
            "the class-names file."
        )


    predicted_index = int(
        np.argmax(predictions)
    )


    predicted_label = CLASS_NAMES[
        predicted_index
    ]


    predicted_confidence = round(
        float(
            predictions[predicted_index]
        ) * 100,
        2
    )


    probabilities = {
        class_name: round(
            float(probability) * 100,
            2
        )

        for class_name, probability
        in zip(
            CLASS_NAMES,
            predictions
        )
    }


    return {
        "weather_label":
            predicted_label,

        "weather_confidence":
            predicted_confidence,

        "weather_probabilities":
            probabilities
    }