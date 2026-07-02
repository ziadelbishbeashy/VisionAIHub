# VisionAIHub

VisionAIHub is a modular AI-powered Flask web application for analyzing road and outdoor images. It combines computer vision, classical machine learning, transfer learning, and a Gemini-based assistant in one platform.

Users can upload an image, select one or more analysis modules, view predictions and confidence values, inspect charts, and ask questions about the combined results.

## Main Features

- YOLOv8 object detection
- Random Forest traffic-density classification
- EfficientNetB0 weather classification
- Transfer-learning scene classification
- Gemini context-aware assistant
- Interactive results dashboard
- Analysis history
- Model confidence and probability visualization
- Modular analysis selection
- Responsive Flask interface

## Analysis Modules

### Object Detection

YOLOv8n detects:

- Person
- Car
- Bus
- Truck
- Motorcycle
- Bicycle
- Traffic light
- Stop sign

Outputs include the annotated image, class counts, total objects, total vehicles, total people, detection confidence, vehicle occupancy, vehicle share, most detected object, and processing time.

### Traffic Intelligence

A Random Forest classifier predicts:

- Low traffic
- Medium traffic
- High traffic

It uses 25 engineered features generated from YOLO outputs, including vehicle counts, occupancy, density, ratios, confidence statistics, and traffic complexity.

Current evaluation:

- Test accuracy: 79.81%
- Weighted F1-score: 72.20%

### Weather Classification

The weather module uses EfficientNetB0 with transfer learning.

Classes:

- Cloudy
- Foggy
- Rainy
- Sunny

Files:

```text
models/weather_model.keras
models/weather_class_names.json
```

The model returns the predicted class, confidence, and probability for every weather class.

### Scene Classification

The scene model predicts:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

It returns the predicted class, confidence, and class probabilities.

### Gemini Assistant

Gemini is the explanation layer, not the prediction layer.

The trained models produce the predictions first. Flask collects their outputs and sends a structured context to Gemini. Gemini then answers questions about objects, traffic, weather, scenes, confidence values, probabilities, and the complete analysis.

## Gemini Working Plan

```text
User asks a question
        ↓
JavaScript sends POST /ask-assistant
        ↓
Flask retrieves the latest analysis context
        ↓
Flask builds structured JSON
        ↓
Context + question are sent to Gemini
        ↓
Gemini generates a grounded explanation
        ↓
Flask returns JSON
        ↓
The answer appears in the chat interface
```

Example structured context:

```json
{
  "object_detection": {
    "total_objects": 12,
    "total_people": 2,
    "total_vehicles": 8
  },
  "traffic_analysis": {
    "prediction": "High",
    "confidence_percent": 84.6
  },
  "weather_analysis": {
    "prediction": "Rainy",
    "confidence_percent": 91.2
  },
  "scene_analysis": {
    "prediction": "Street",
    "confidence_percent": 88.4
  }
}
```

Gemini prompt rules:

- Use only the supplied model results
- Never invent objects, counts, predictions, or confidence values
- State when a module was not selected or failed
- Do not claim the traffic-light color
- Do not guarantee road safety
- Mention low-confidence predictions
- Answer in the same language as the user

Reliability features:

- Automatic retry for temporary server errors
- Backup Gemini model support
- Clear error messages
- Backend-only API access
- Environment-variable API key storage

## System Architecture

```text
User Layer
    ↓
Flask Web Application
    ↓
Selected AI Modules
    ├── YOLOv8 Object Detection
    ├── Random Forest Traffic Classification
    ├── EfficientNetB0 Weather Classification
    └── Scene Transfer-Learning Model
    ↓
Results Aggregation
    ↓
Dashboard, Charts, History, and Gemini Assistant
```

Suggested architecture image:

```text
static/images/visionaihub_system_architecture_diagram.png
```

## Technologies Used

### Frontend

- HTML5
- CSS3
- JavaScript
- Jinja2
- Chart.js

### Backend

- Python
- Flask
- Werkzeug
- Flask Session

### AI and Machine Learning

- Ultralytics YOLOv8
- TensorFlow
- Keras
- Scikit-learn
- EfficientNetB0
- Random Forest
- Transfer learning
- Feature engineering
- Google Gemini API

### Data and Utilities

- Pandas
- NumPy
- Joblib
- JSON
- CSV

### Development Tools

- Visual Studio Code
- Google Colab
- Kaggle
- Git
- GitHub

## Project Structure

```text
VisionAIHub/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── weather_model.keras
│   ├── weather_class_names.json
│   ├── scene_model.keras
│   └── ml/
│       ├── traffic_density_model.pkl
│       └── traffic_features.pkl
│
├── utils/
│   ├── detection.py
│   ├── ml_predictor.py
│   ├── weather_classifier.py
│   ├── scene_classifier.py
│   └── gemini_assistant.py
│
├── templates/
│   ├── index.html
│   ├── results.html
│   ├── history.html
│   └── model_performance.html
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── js/results_charts.js
│   ├── uploads/
│   ├── results/
│   ├── model_plots/
│   └── images/
│
└── data/
    ├── object_counts.csv
    ├── analysis_history.csv
    └── model_metrics.json
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd VisionAIHub
```

Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Important packages:

```text
Flask
tensorflow
ultralytics
scikit-learn
pandas
numpy
joblib
google-genai
```

## Gemini API Setup

Store the API key in an environment variable.

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Windows Command Prompt:

```bat
set GEMINI_API_KEY=YOUR_API_KEY
```

Permanent Windows variable:

```bat
setx GEMINI_API_KEY "YOUR_API_KEY"
```

Never place the API key in Python, HTML, JavaScript, GitHub, or screenshots.

## Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How to Use

1. Upload an image.
2. Select one or more analysis modules.
3. Set the YOLO confidence threshold when required.
4. Run the analysis.
5. View prediction cards, probabilities, object counts, and charts.
6. Ask Gemini questions about the available results.
7. Open history or model-performance pages when available.

## Analysis Presets

- Full Analysis
- Traffic Only
- Weather Only
- Scene Only
- Detection Only

Traffic Only also enables YOLO because traffic features depend on object detections.

## Visual Analytics

The result page can display:

- Object-count bar chart
- People-versus-vehicles doughnut chart
- Traffic class-probability chart
- Weather class-probability chart
- Scene class-probability chart
- Model-confidence comparison chart

The model-performance page can display:

- Confusion matrices
- Accuracy curves
- Loss curves
- Feature importance
- Precision, recall, and F1-score
- Model-comparison charts

## Model Performance

### Random Forest

- Accuracy: 72.81%
- Weighted F1-score: 72.20%
- Features: 25

### Weather Model

Add the final retrained results:

```text
Test accuracy:
Weighted precision:
Weighted recall:
Weighted F1-score:
```

### Scene Model

Add the final results:

```text
Test accuracy:
Weighted precision:
Weighted recall:
Weighted F1-score:
```

## Security

- API keys are stored in environment variables
- `.env` is excluded with `.gitignore`
- Generated uploads and results can be excluded from GitHub
- Uploaded filenames are sanitized
- Gemini requests are sent only from Flask

Recommended `.gitignore`:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
static/uploads/*
static/results/*
```

## Limitations

- Results depend on image quality and lighting
- Small or hidden objects may be missed
- Traffic classification depends on YOLO outputs
- One static image cannot represent full real-time road conditions
- Weather and scene predictions are limited to trained classes
- Gemini explanations depend on the supplied context
- Gemini may temporarily return service errors during high demand

## Future Improvements

- Real-time video analysis
- Live traffic-camera integration
- Larger and more diverse datasets
- Improved weather and scene models
- Grad-CAM explainability
- Unified database history
- Full analytics dashboard
- Downloadable PDF reports
- User authentication
- Cloud deployment
- Multilingual assistant support

## Team

- Ziad Elbishbeashy
- Nour Diaa
- Maya Bassem

## Important Note

VisionAIHub provides AI-assisted image analysis. It should not replace official traffic, weather, or safety systems.
