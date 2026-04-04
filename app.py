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

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'phishing_model.pkl')

# Load model once at startup
model = None

def load_model():
    """Load the Random Forest model from disk."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run 'python model/train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")

# ── Helpers ───────────────────────────────────────────────────────────────────

URL_REGEX = re.compile(
    r'^(https?://)'           # must start with http:// or https://
    r'[^\s/$.?#].[^\s]*$',   # followed by a non-whitespace domain + path
    re.IGNORECASE
)

def validate_url(url: str) -> tuple[bool, str]:
    """
    Basic URL validation.
    Returns (is_valid: bool, error_message: str).
    """
    url = url.strip()
    if not url:
        return False, "URL cannot be empty."
    if len(url) > 2048:
        return False, "URL is too long (max 2048 characters)."
    if not URL_REGEX.match(url):
        return False, "Invalid URL. Please include http:// or https://"
    return True, ""


def predict_url(url: str) -> dict:
    """
    Run inference on a single URL.
    Returns a result dict with prediction, confidence, and features.
    """
    if model is None:
        raise RuntimeError("Model is not loaded.")

    # Extract features
    features = extract_features(url)
    feature_desc = get_feature_description(url)

    X = np.array([features])

    # Predict
    prediction  = int(model.predict(X)[0])          # 0 or 1
    probabilities = model.predict_proba(X)[0]        # [P(legit), P(phish)]

    confidence  = float(max(probabilities)) * 100
    phish_prob  = float(probabilities[1]) * 100
    legit_prob  = float(probabilities[0]) * 100

    label  = "Phishing" if prediction == 1 else "Legitimate"
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
        "url":           url,
        "prediction":    prediction,
        "label":         label,
        "is_safe":       is_safe,
        "confidence":    round(confidence, 2),
        "phish_prob":    round(phish_prob, 2),
        "legit_prob":    round(legit_prob, 2),
        "risk_level":    risk_level,
        "features":      feature_desc,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    Accepts form POST or JSON body.
    Returns JSON result.
    """
    # Support both form data and JSON
    if request.is_json:
        data = request.get_json()
        url  = data.get('url', '').strip()
    else:
        url = request.form.get('url', '').strip()

    # Validate
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        return jsonify({
            "success": False,
            "error":   error_msg
        }), 400

    try:
        result = predict_url(url)
        return jsonify({
            "success": True,
            **result
        })
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        app.logger.error(f"Prediction error for '{url}': {e}")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred during prediction."
        }), 500


@app.route('/health')
def health():
    """Health-check endpoint."""
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO]  Training model now...\n")
        os.system(f'python {os.path.join(os.path.dirname(__file__), "model", "train_model.py")}')
        load_model()

    print("\n[INFO] Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
