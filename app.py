"""
AgriScan AI - Crop Disease Detection System
app.py — Main Flask backend file

This file does the following:
  1. Loads your two trained Keras models (potato + cassava)
  2. Accepts an uploaded image (or camera capture) from the browser
  3. Preprocesses the image the same way your models were trained
  4. Runs a prediction and returns the disease name + confidence score
  5. Serves the HTML page at http://127.0.0.1:5000
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os                          # For file-path operations
import io                          # For handling image bytes in memory
import base64                      # For decoding camera images sent as Base64
import textwrap                    # For wrapping text in PDF report
from datetime import datetime
import numpy as np                 # Numerical processing (model input arrays)

from flask import (
    Flask,
    render_template,               # Renders your HTML template
    request,                       # Reads form data & uploaded files
    jsonify,                       # Returns JSON responses to the browser
    send_file                      # Sends generated PDF back to the browser
)
from PIL import Image              # Pillow — opens and resizes images
import tensorflow as tf            # TensorFlow / Keras for loading models
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ─── Flask App Initialisation ─────────────────────────────────────────────────
app = Flask(__name__)

# Maximum allowed upload size: 5 MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB in bytes

# ─── Model Paths ──────────────────────────────────────────────────────────────
# Place your .keras model files inside a folder called "models/"
POTATO_MODEL_PATH  = os.path.join("models", "potato_final_model.keras")
CASSAVA_MODEL_PATH = os.path.join("models", "cassava_model.keras")

# ─── Class Labels ─────────────────────────────────────────────────────────────
# These MUST match the order your models were trained on.
# Keras sorts class folders alphabetically during training, so keep this order.
POTATO_CLASSES  = ["Late Blight", "Early Blight", "Healthy"]
CASSAVA_CLASSES = [
    "Bacterial Blight",
    "Healthy",
    "Mosaic Disease",
]

# ─── Disease Suggestions ──────────────────────────────────────────────────────
# Short treatment / management advice shown to the user after prediction.
SUGGESTIONS = {
    "Early Blight": (
        "Early Blight is caused by the fungus Alternaria solani. "
        "Apply copper-based or chlorothalonil fungicides every 7–10 days. "
        "Remove and destroy infected leaves. Avoid overhead irrigation."
    ),
    "Late Blight": (
        "Late Blight is caused by Phytophthora infestans. "
        "Apply fungicides containing mancozeb or metalaxyl immediately. "
        "Destroy infected plant material. Ensure good field drainage."
    ),
    "Healthy": (
        "Your crop appears healthy! Continue regular monitoring, "
        "maintain proper spacing for air circulation, and follow a "
        "preventive spray schedule during wet seasons."
    ),
    "Bacterial Blight": (
        "Cassava Bacterial Blight (CBB) is caused by Xanthomonas axonopodis. "
        "Use disease-free planting material, practice crop rotation, "
        "and remove infected plants promptly to stop spread."
    ),
    "Mosaic Disease": (
        "Cassava Mosaic Disease is spread by whiteflies. "
        "Use resistant varieties, control whitefly populations with "
        "insecticides or neem oil, and remove infected plants immediately."
    ),
}

# ─── Load Models Once at Startup ──────────────────────────────────────────────
# We load both models when Flask starts so predictions are instant.
print("⏳ Loading models — please wait...")

potato_model  = None
cassava_model = None

if os.path.exists(POTATO_MODEL_PATH):
    potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
    print(f"✅ Potato model loaded from  '{POTATO_MODEL_PATH}'")
else:
    print(f"⚠️  Potato model NOT found at '{POTATO_MODEL_PATH}' — predictions will fail.")

if os.path.exists(CASSAVA_MODEL_PATH):
    cassava_model = tf.keras.models.load_model(CASSAVA_MODEL_PATH)
    print(f"✅ Cassava model loaded from '{CASSAVA_MODEL_PATH}'")
else:
    print(f"⚠️  Cassava model NOT found at '{CASSAVA_MODEL_PATH}' — predictions will fail.")

print("🚀 Server ready!\n")


# ─── Helper: Preprocess Image ─────────────────────────────────────────────────
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Resizes and normalises a PIL image into a NumPy array that the model
    expects as input.

    Steps:
      1. Convert to RGB (removes alpha channel if PNG has one)
      2. Resize to 224×224 (standard for MobileNet / EfficientNet / CNN)
      3. Convert to float32 NumPy array with shape (224, 224, 3)
      4. Normalise pixel values from [0, 255] → [0.0, 1.0]
      5. Add a batch dimension → shape becomes (1, 224, 224, 3)
         because Keras models always expect a batch, even for one image.

    If you trained on a different image size, change target_size here.
    If you used tf.keras.applications.preprocess_input() during training,
    replace the /255.0 line with that function instead.
    """
    image = image.convert("RGB")                          # Ensure 3-channel RGB
    image = image.resize(target_size)                     # Resize to model's expected input
    img_array = np.array(image, dtype=np.float32)         # Convert to NumPy float array
    img_array = img_array / 255.0                         # Normalise to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)         # Add batch dimension: (1, 224, 224, 3)
    return img_array


# ─── Helper: Run Prediction ───────────────────────────────────────────────────
def predict(model, class_labels: list, img_array: np.ndarray) -> dict:
    """
    Runs a forward pass through the model and returns the predicted
    class name and confidence score.

    model        — loaded Keras model
    class_labels — list of string class names in training order
    img_array    — preprocessed NumPy array with shape (1, H, W, 3)

    Returns a dict: { "disease": str, "confidence": float (0–100) }
    """
    predictions = model.predict(img_array)          # Shape: (1, num_classes)
    class_index = int(np.argmax(predictions[0]))    # Index of highest probability
    confidence  = float(np.max(predictions[0]))     # The highest probability value

    disease_name = class_labels[class_index]        # Map index → class name
    confidence_pct = round(confidence * 100, 2)     # Convert to percentage

    return {
        "disease":    disease_name,
        "confidence": confidence_pct,
        "suggestion": SUGGESTIONS.get(disease_name, "Consult an agricultural expert.")
    }


# ─── Route: Home Page ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    """
    Serves the main HTML page.
    Flask looks for 'index.html' inside the 'templates/' folder automatically.
    """
    return render_template("index.html")


# ─── Route: Predict (File Upload) ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handles image upload from the browser form.

    Expects a multipart/form-data POST with:
      - 'crop'  : "potato" or "cassava"
      - 'image' : the uploaded image file

    Returns JSON:
      { "disease": str, "confidence": float, "suggestion": str }
    or
      { "error": str }  on failure
    """
    # ── 1. Read form fields ──────────────────────────────────────────────────
    crop_type = request.form.get("crop", "").lower().strip()
    image_file = request.files.get("image")

    # ── 2. Validate inputs ───────────────────────────────────────────────────
    if not crop_type:
        return jsonify({"error": "Please select a crop type."}), 400

    if image_file is None or image_file.filename == "":
        return jsonify({"error": "No image was uploaded."}), 400

    allowed_extensions = {"jpg", "jpeg", "png", "webp", "jfif"}
    ext = image_file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 400

    # ── 3. Choose model & classes ────────────────────────────────────────────
    if crop_type == "potato":
        if potato_model is None:
            return jsonify({"error": "Potato model not loaded. Check server logs."}), 500
        model  = potato_model
        labels = POTATO_CLASSES
    elif crop_type == "cassava":
        if cassava_model is None:
            return jsonify({"error": "Cassava model not loaded. Check server logs."}), 500
        model  = cassava_model
        labels = CASSAVA_CLASSES
    else:
        return jsonify({"error": f"Unknown crop type: '{crop_type}'."}), 400

    # ── 4. Open & preprocess image ───────────────────────────────────────────
    try:
        image = Image.open(image_file.stream)      # Open the uploaded file
        img_array = preprocess_image(image)        # Resize + normalise
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    # ── 5. Predict & return result ───────────────────────────────────────────
    try:
        result = predict(model, labels, img_array)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─── Route: Predict from Camera (Base64) ─────────────────────────────────────
@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    """
    Handles camera images sent from the browser as Base64-encoded strings.

    The browser captures a frame from the webcam, converts it to a
    Base64 data URL (e.g. "data:image/png;base64,iVBOR..."), and POSTs
    it here as JSON:
      { "crop": "potato", "image": "data:image/png;base64,..." }

    We decode the Base64 back to bytes, open it with Pillow, then predict.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received."}), 400

    crop_type    = data.get("crop", "").lower().strip()
    image_data   = data.get("image", "")

    # ── Validate ─────────────────────────────────────────────────────────────
    if not crop_type:
        return jsonify({"error": "Please select a crop type."}), 400
    if not image_data:
        return jsonify({"error": "No camera image received."}), 400

    # ── Choose model ─────────────────────────────────────────────────────────
    if crop_type == "potato":
        if potato_model is None:
            return jsonify({"error": "Potato model not loaded."}), 500
        model, labels = potato_model, POTATO_CLASSES
    elif crop_type == "cassava":
        if cassava_model is None:
            return jsonify({"error": "Cassava model not loaded."}), 500
        model, labels = cassava_model, CASSAVA_CLASSES
    else:
        return jsonify({"error": f"Unknown crop type: '{crop_type}'."}), 400

    # ── Decode Base64 image ───────────────────────────────────────────────────
    try:
        # Strip the "data:image/...;base64," prefix
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        img_array = preprocess_image(image)
    except Exception as e:
        return jsonify({"error": f"Could not decode camera image: {str(e)}"}), 400

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predict(model, labels, img_array)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        

# ─── Run ──────────────────────────────────────────────────────────────────────

@app.route("/download_report", methods=["POST"])
def download_report():
    """Generate a PDF report from the latest diagnosis and return it."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid report request."}), 400

    crop_type = data.get("crop", "").capitalize()
    disease = data.get("disease", "")
    confidence = data.get("confidence")
    suggestion = data.get("suggestion", "")

    if not crop_type or not disease or confidence is None or not suggestion:
        return jsonify({"error": "Missing required report fields."}), 400

    try:
        buffer = io.BytesIO()
        page_width, page_height = letter
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("AgriScan AI Diagnostic Report")

        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(50, page_height - 70, "AgriScan AI")

        pdf.setFont("Helvetica", 12)
        timestamp = datetime.now().strftime("%B %d, %Y %H:%M")
        pdf.drawString(50, page_height - 100, f"Report date: {timestamp}")
        pdf.drawString(50, page_height - 120, f"Crop type: {crop_type}")
        pdf.drawString(50, page_height - 140, f"Predicted disease: {disease}")
        pdf.drawString(50, page_height - 160, f"Confidence: {confidence}%")

        pdf.line(50, page_height - 170, page_width - 50, page_height - 170)

        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, page_height - 190, "AI suggestion / treatment")

        pdf.setFont("Helvetica", 11)
        wrapped_suggestion = textwrap.wrap(suggestion, width=80)
        text_object = pdf.beginText(50, page_height - 210)
        text_object.setLeading(16)
        for line in wrapped_suggestion:
            text_object.textLine(line)
        pdf.drawText(text_object)

        pdf.showPage()
        pdf.save()
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="AgriScan_Report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"error": f"Could not generate report: {str(e)}"}), 500

if __name__ == "__main__":
    # debug=True: Flask reloads automatically when you save app.py
    # Remove debug=True when presenting / deploying
    app.run(debug=True, port=5000)