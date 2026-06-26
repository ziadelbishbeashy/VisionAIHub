from flask import Flask, render_template, request
import os
from utils.detection import detect_objects

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    result_path, counts = detect_objects(image_path, RESULT_FOLDER)

    return render_template(
        "results.html",
        original_image=image_path,
        result_image=result_path,
        counts=counts
    )

if __name__ == "__main__":
    app.run(debug=True)