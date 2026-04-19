import os
import io
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Logging ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────
# CONFIG
# ───────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet18_disease_model.pth")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 8))

# ⚠️ IMPORTANT: Replace this with EXACT order from training
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
# MODEL (LAZY LOAD)
# ───────────────────────────────────────
model = None

def load_model():
    logger.info("🔄 Loading model...")
    m = models.resnet18(weights=None)
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()

    logger.info("✅ Model loaded")
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
# ROUTES
# ───────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "DL Model API Running",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        logger.info("📥 Request received")

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Load image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Transform
        input_tensor = transform(image).unsqueeze(0)

        # Load model
        model = get_model()
        logger.info("🤖 Model ready")

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        # 🔥 DEBUG LOGS
        logger.info(f"📊 Prediction index: {idx.item()}")
        logger.info(f"📊 Probabilities: {probs.tolist()}")

        predicted = CLASS_NAMES[idx.item()]

        logger.info(f"✅ Final prediction: {predicted}")

        return jsonify({
            "predicted_disease": predicted,
            "confidence": round(confidence.item() * 100, 2),

            # 🔥 Debug (remove later if needed)
            "debug_index": idx.item(),
            "debug_probs": probs.tolist()
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
    logger.info(f"🚀 Server starting on port {port}")
    app.run(host="0.0.0.0", port=port)
