from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess
)

from tensorflow.keras.utils import (
    img_to_array,
    load_img
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "scene_model.keras"
)


# This order must match the training dataset exactly.
SCENE_CLASS_NAMES = [
    "Buildings",
    "Forest",
    "Glacier",
    "Mountain",
    "Sea",
    "Street"
]


# Available modes:
# "auto"          -> use model Rescaling layer if included;
#                    otherwise divide pixels by 255
# "rescale_01"    -> pixels become 0 to 1
# "mobilenet_v2"  -> MobileNetV2 preprocess_input
# "efficientnet"  -> EfficientNet preprocess_input
# "vgg16"         -> VGG16 preprocess_input
#
# Start with "auto". Change it only when your training
# notebook explicitly used another preprocess_input.
PREPROCESSING_MODE = "auto"


@lru_cache(maxsize=1)
def get_scene_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Scene model was not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


def model_contains_rescaling(model):
    """
    Check whether the saved model already contains a
    Keras Rescaling layer.
    """

    def inspect_layers(layers):
        for layer in layers:
            if (
                layer.__class__.__name__.lower()
                == "rescaling"
            ):
                return True

            nested_layers = getattr(
                layer,
                "layers",
                None
            )

            if (
                nested_layers
                and inspect_layers(nested_layers)
            ):
                return True

        return False

    return inspect_layers(model.layers)


def get_model_image_size(model):
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected scene model input shape: "
            f"{input_shape}"
        )

    image_height = input_shape[1]
    image_width = input_shape[2]

    # Fallback for models with dynamic image dimensions.
    if image_height is None:
        image_height = 224

    if image_width is None:
        image_width = 224

    return int(image_height), int(image_width)


def preprocess_scene_array(
    image_array,
    model
):
    image_array = image_array.astype(
        np.float32
    )

    if PREPROCESSING_MODE == "mobilenet_v2":
        return mobilenet_preprocess(
            image_array
        )

    if PREPROCESSING_MODE == "efficientnet":
        return efficientnet_preprocess(
            image_array
        )

    if PREPROCESSING_MODE == "vgg16":
        return vgg16_preprocess(
            image_array
        )

    if PREPROCESSING_MODE == "rescale_01":
        return image_array / 255.0

    if PREPROCESSING_MODE == "auto":
        # When the model already contains Rescaling,
        # send normal 0–255 pixels to the model.
        if model_contains_rescaling(model):
            return image_array

        # Common image_dataset_from_directory setup.
        return image_array / 255.0

    raise ValueError(
        "Invalid PREPROCESSING_MODE: "
        f"{PREPROCESSING_MODE}"
    )


def normalize_predictions(raw_predictions):
    """
    Supports models whose final layer returns either
    probabilities or raw logits.
    """

    values = np.asarray(
        raw_predictions,
        dtype=np.float32
    ).reshape(-1)

    probability_sum = float(
        np.sum(values)
    )

    already_probabilities = (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(
            probability_sum,
            1.0,
            atol=0.01
        )
    )

    if already_probabilities:
        return values

    return tf.nn.softmax(
        values
    ).numpy()


def predict_scene(image_path):
    model = get_scene_model()

    image_height, image_width = (
        get_model_image_size(model)
    )

    image = load_img(
        image_path,
        target_size=(
            image_height,
            image_width
        ),
        color_mode="rgb"
    )

    image_array = img_to_array(image)

    image_array = preprocess_scene_array(
        image_array,
        model
    )

    batch = np.expand_dims(
        image_array,
        axis=0
    )

    raw_output = model.predict(
        batch,
        verbose=0
    )

    if isinstance(raw_output, list):
        raw_output = raw_output[0]

    raw_output = np.asarray(raw_output)

    if raw_output.ndim > 1:
        raw_output = raw_output[0]

    probabilities = normalize_predictions(
        raw_output
    )

    if (
        len(probabilities)
        != len(SCENE_CLASS_NAMES)
    ):
        raise ValueError(
            "Scene model output count does not match "
            "the number of class names. "
            f"Model outputs: {len(probabilities)}, "
            f"class names: {len(SCENE_CLASS_NAMES)}"
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = (
        SCENE_CLASS_NAMES[predicted_index]
    )

    confidence = round(
        float(
            probabilities[predicted_index]
        ) * 100,
        2
    )

    scene_probabilities = {
        class_name: round(
            float(probability) * 100,
            2
        )
        for class_name, probability in zip(
            SCENE_CLASS_NAMES,
            probabilities
        )
    }

    return {
        "scene_label": predicted_label,
        "scene_confidence": confidence,
        "scene_probabilities": scene_probabilities,
        "scene_input_width": image_width,
        "scene_input_height": image_height,
        "scene_model_name": model.name
    }