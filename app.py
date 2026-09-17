# app.py
# Flask backend for AI-Based Phishing Website Detection System

import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# Import our feature extractor
from utils.feature_extraction import extract_features, get_feature_description


# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)


# ── Model setup ──────────────────────────────────────────────────────────────

# Path to the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "phishing_model.pkl"
)

# Model variable
model = None


def load_model():
    """Load the Random Forest model from disk."""
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Make sure model/phishing_model.pkl exists."
        )

    model = joblib.load(MODEL_PATH)

    print(f"[INFO] Model loaded from: {MODEL_PATH}")


# Load the model when Flask/Gunicorn starts.
# This is important for Render deployment.
try:
    load_model()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None


# ── URL validation ──────────────────────────────────────────────────────────

URL_REGEX = re.compile(
    r"^(https?://)"
    r"[^\s/$.?#].[^\s]*$",
    re.IGNORECASE
)


def validate_url(url: str) -> tuple[bool, str]:
    """
    Basic URL validation.

    Returns:
        (is_valid, error_message)
    """

    url = url.strip()

    if not url:
        return False, "URL cannot be empty."

    if len(url) > 2048:
        return False, "URL is too long (max 2048 characters)."

    if not URL_REGEX.match(url):
        return False, "Invalid URL. Please include http:// or https://"

    return True, ""


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_url(url: str) -> dict:
    """
    Run inference on a single URL.

    Returns a result dictionary containing:
    prediction, confidence, probabilities, risk level and features.
    """

    if model is None:
        raise RuntimeError("Model is not loaded.")

    # Extract features
    features = extract_features(url)
    feature_desc = get_feature_description(url)

    # Convert features to numpy array
    X = np.array([features])

    # Prediction
    prediction = int(model.predict(X)[0])

    # Probability:
    # [P(legitimate), P(phishing)]
    probabilities = model.predict_proba(X)[0]

    confidence = float(max(probabilities)) * 100
    phish_prob = float(probabilities[1]) * 100
    legit_prob = float(probabilities[0]) * 100

    # Label
    label = "Phishing" if prediction == 1 else "Legitimate"
    is_safe = prediction == 0

    # Risk level based on phishing probability
    if phish_prob >= 80:
        risk_level = "High"
    elif phish_prob >= 50:
        risk_level = "Medium"
    elif phish_prob >= 25:
        risk_level = "Low"
    else:
        risk_level = "Very Low"

    return {
        "url": url,
        "prediction": prediction,
        "label": label,
        "is_safe": is_safe,
        "confidence": round(confidence, 2),
        "phish_prob": round(phish_prob, 2),
        "legit_prob": round(legit_prob, 2),
        "risk_level": risk_level,
        "features": feature_desc,
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the homepage."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests.

    Accepts:
    - Form POST
    - JSON body

    Returns:
    - JSON result
    """

    # Support both form data and JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
        url = data.get("url", "").strip()
    else:
        url = request.form.get("url", "").strip()

    # Validate URL
    is_valid, error_msg = validate_url(url)

    if not is_valid:
        return jsonify({
            "success": False,
            "error": error_msg
        }), 400

    try:
        result = predict_url(url)

        return jsonify({
            "success": True,
            **result
        })

    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 503

    except Exception as e:
        app.logger.error(
            f"Prediction error for '{url}': {e}"
        )

        return jsonify({
            "success": False,
            "error": "An unexpected error occurred during prediction."
        }), 500


@app.route("/health")
def health():
    """Health-check endpoint."""

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[INFO] Starting Flask development server...")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )