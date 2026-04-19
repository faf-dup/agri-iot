import os
import io
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ───────────────────────────────────────
# LOGGING
# ───────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────
# CONFIG
# ───────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet18_disease_model.pth")
NUM_CLASSES = 8

# ⚠ MUST match training order
CLASS_NAMES = [
    "Apple__Apple_scab",
    "Apple__Black_rot",
    "Apple__Cedar_apple_rust",
    "Apple__healthy",
    "Corn_(maize)__Northern_Leaf_Blight",
    "Corn_(maize)__healthy",
    "Grape__Black_rot",
    "Grape__Esca(Black_Measles)"
]

# ───────────────────────────────────────
# DEVICE
# ───────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────
# MODEL (LAZY LOAD)
# ───────────────────────────────────────
model = None

def load_model():
    logger.info("🔄 Loading model...")

    m = models.resnet18(weights=None)
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device)
    m.eval()

    logger.info("✅ Model loaded successfully")
    return m

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# ───────────────────────────────────────
# TRANSFORM (MATCH TRAINING)
# ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ───────────────────────────────────────
# TREATMENT DB
# ───────────────────────────────────────
TREATMENT_DB = {

    "Apple__Apple_scab": {
        "status": "Diseased 🔴",
        "description": "Fungal infection causing olive-green to dark lesions on leaves and fruits.",
        "treatment": "Apply fungicides like captan or myclobutanil. Remove infected leaves and improve air circulation."
    },

    "Apple__Black_rot": {
        "status": "Diseased 🔴",
        "description": "Dark rot on fruits and spots on leaves and branches.",
        "treatment": "Prune infected areas and apply copper-based fungicides."
    },

    "Apple__Cedar_apple_rust": {
        "status": "Diseased 🔴",
        "description": "Rust disease causing yellow-orange spots on leaves.",
        "treatment": "Use fungicides like propiconazole and avoid cedar trees nearby."
    },

    "Apple__healthy": {
        "status": "Healthy 🟢",
        "description": "No disease symptoms detected.",
        "treatment": "Maintain regular watering and fertilization."
    },

    "Corn_(maize)__Northern_Leaf_Blight": {
        "status": "Diseased 🔴",
        "description": "Gray-green lesions on leaves reducing yield.",
        "treatment": "Use resistant varieties and apply fungicides early."
    },

    "Corn_(maize)__healthy": {
        "status": "Healthy 🟢",
        "description": "Plant is healthy.",
        "treatment": "Continue proper nutrient and water management."
    },

    "Grape__Black_rot": {
        "status": "Diseased 🔴",
        "description": "Dark lesions and shriveled fruits.",
        "treatment": "Apply fungicides like mancozeb and remove infected debris."
    },

    "Grape__Esca(Black_Measles)": {
        "status": "Diseased 🔴",
        "description": "Leaf discoloration and vine decline.",
        "treatment": "Prune infected wood. No complete cure available."
    }
}

# ───────────────────────────────────────
# ROUTES
# ───────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "message": "DL API Running",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/predict_disease", methods=["POST"])
def predict():
    try:
        logger.info("📥 Request received")

        if "file" not in request.files:
            return jsonify({"error": "Upload image using key 'file'"}), 400

        file = request.files["file"]

        # preprocess image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        model = get_model()

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        predicted_class = CLASS_NAMES[idx.item()]
        confidence = round(confidence.item() * 100, 2)

        logger.info(f"✅ Prediction: {predicted_class} ({confidence}%)")

        treatment_info = TREATMENT_DB.get(predicted_class, {})

        return jsonify({
            "predicted_disease": predicted_class,   # ✅ same as before
            "confidence": confidence,
            "status": treatment_info.get("status"),
            "description": treatment_info.get("description"),
            "treatment": treatment_info.get("treatment")
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
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
