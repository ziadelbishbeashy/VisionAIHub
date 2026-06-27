from flask import Flask, render_template, request
import os
import pandas as pd
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
    confidence = float(request.form.get("confidence", 0.4))

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    result_path, summary = detect_objects(
        image_path,
        RESULT_FOLDER,
        confidence_threshold=confidence
    )

    return render_template(
        "results.html",
        original_image=image_path,
        result_image=result_path,
        summary=summary,
        confidence=confidence
    )
@app.route("/history")
def history():
    csv_path = "data/object_counts.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        records = df.tail(20).iloc[::-1].to_dict(orient="records")
    else:
        records = []

    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)