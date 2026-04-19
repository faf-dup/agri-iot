import os
import io
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Logging ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Allow all origins (easy for now)
CORS(app)

# ───────────────────────────────────────
# CONFIG
# ───────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet18_disease_model.pth")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 8))
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "class_names.json")

# ───────────────────────────────────────
# LOAD CLASS NAMES
# ───────────────────────────────────────
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [
        "Healthy",
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Leaf Smut",
        "Blast",
        "Tungro",
        "Sheath Blight",
        "False Smut",
    ]

# ───────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    logger.info("✅ ResNet18 model loaded")
    return model

model = load_model()

# ───────────────────────────────────────
# IMAGE TRANSFORM
# ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ───────────────────────────────────────
# ROUTES
# ───────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "DL Model API Running",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        disease = CLASS_NAMES[idx.item()]

        return jsonify({
            "predicted_disease": disease,
            "confidence": round(confidence.item() * 100, 2)
        })

    except Exception as e:
        logger.error("❌ Error: %s", str(e))
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e)
        }), 500


# ───────────────────────────────────────
# RUN
# ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
