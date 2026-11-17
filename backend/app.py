
import os
import json
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "waste_classifier.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

# Load trained CNN model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("CNN Model loaded successfully!")
print("Classes:", class_names)


@app.route("/")
def home():
    return jsonify({
        "message": "EcoVision AI Backend is running!"
    })


@app.route("/predict/waste", methods=["POST"])
def predict_waste():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]

        # Read and preprocess image
        image = Image.open(file).convert("RGB")
        image = image.resize((128, 128))

        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        return jsonify({
            "success": True,
            "waste_type": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
