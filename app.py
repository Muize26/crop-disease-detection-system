"""
Crop Disease Detection System
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
import threading                   # Prevents duplicate model loads during concurrent requests
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np                 # Numerical processing (model input arrays)

from flask import (
    Flask,
    render_template,               # Renders your HTML template
    request,                       # Reads form data & uploaded files
    jsonify,                       # Returns JSON responses to the browser
    send_file                      # Sends generated PDF back to the browser
)
from PIL import Image, ImageOps
import tensorflow as tf            # TensorFlow / Keras for loading models
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

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

# ─── Lazy Model Cache ────────────────────────────────────────────────────────
# Models are loaded only on the first prediction request for the selected crop,
# then cached in memory for later requests. This keeps Render startup lean and
# avoids loading both large models at once.
MODEL_CACHE = {}
MODEL_LOCK = threading.Lock()


def _configure_tensorflow_memory() -> None:
    """Avoid TensorFlow grabbing all available memory at once."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


_configure_tensorflow_memory()

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

# ─── Lazy Model Loader ──────────────────────────────────────────────────────
def get_model_for_crop(crop_type: str) -> tuple:
    """Load a model only when first needed and cache it globally."""
    normalized_crop = crop_type.lower().strip()

    if normalized_crop not in {"potato", "cassava"}:
        raise ValueError(f"Unknown crop type: '{crop_type}'.")

    if normalized_crop in MODEL_CACHE:
        return MODEL_CACHE[normalized_crop]["model"], MODEL_CACHE[normalized_crop]["labels"]

    with MODEL_LOCK:
        if normalized_crop in MODEL_CACHE:
            return MODEL_CACHE[normalized_crop]["model"], MODEL_CACHE[normalized_crop]["labels"]

        if normalized_crop == "potato":
            model_path = POTATO_MODEL_PATH
            labels = POTATO_CLASSES
        else:
            model_path = CASSAVA_MODEL_PATH
            labels = CASSAVA_CLASSES

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"⏳ Loading {normalized_crop} model on first request...")
        model = tf.keras.models.load_model(model_path, compile=False)
        MODEL_CACHE[normalized_crop] = {"model": model, "labels": labels}
        print(f"✅ {normalized_crop.capitalize()} model cached for future predictions.")
        return model, labels


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
    img_array = np.asarray(image, dtype=np.float32)       # Convert to NumPy float array
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
    predictions = model.predict(img_array, verbose=0)  # Shape: (1, num_classes)
    class_index = int(np.argmax(predictions[0]))       # Index of highest probability
    confidence  = float(np.max(predictions[0]))        # The highest probability value

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

    # ── 3. Load model lazily ────────────────────────────────────────────────
    try:
        model, labels = get_model_for_crop(crop_type)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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

    # ── Load model lazily ───────────────────────────────────────────────────
    try:
        model, labels = get_model_for_crop(crop_type)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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
    image_data = data.get("image_data", "")

    if not crop_type or not disease or confidence is None or not suggestion:
        return jsonify({"error": "Missing required report fields."}), 400

    report_image = None
    if image_data:
        try:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
            report_image = Image.open(io.BytesIO(image_bytes))
            report_image = ImageOps.exif_transpose(report_image)
            if report_image.mode not in ("RGB", "L"):
                report_image = report_image.convert("RGB")
        except Exception:
            report_image = None

    try:
        buffer = io.BytesIO()
        page_width, page_height = letter
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("AgriScan AI Diagnostic Report")

        pdf.setFont("Helvetica-Bold", 26)
        pdf.setFillColorRGB(0.10, 0.40, 0.18)
        pdf.drawString(50, page_height - 70, "AgriScan AI")

        pdf.setFont("Helvetica", 12)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(50, page_height - 98, "AI Crop Disease Diagnostic Report")

        pdf.setStrokeColorRGB(0.78, 0.78, 0.78)
        pdf.setLineWidth(1)
        pdf.line(50, page_height - 108, page_width - 50, page_height - 108)

        left_x = 50
        current_y = page_height - 140
        line_height = 18

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left_x, current_y, "Date & Time")
        pdf.setFont("Helvetica", 10)
        timestamp = datetime.now().strftime("%B %d, %Y %H:%M")
        pdf.drawString(left_x, current_y - line_height, timestamp)

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left_x, current_y - 2 * line_height - 4, "Crop Type")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left_x, current_y - 3 * line_height - 4, crop_type)

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left_x, current_y - 4 * line_height - 10, "Predicted Disease")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left_x, current_y - 5 * line_height - 10, disease)

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left_x, current_y - 6 * line_height - 16, "Confidence")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left_x, current_y - 7 * line_height - 16, f"{confidence}%")

        pdf.setStrokeColorRGB(0.85, 0.85, 0.85)
        pdf.roundRect(45, current_y - 7 * line_height - 34, page_width - 90, 140, radius=10, stroke=1, fill=0)

        pdf.setFont("Helvetica-Bold", 12)
        pdf.setFillColorRGB(0.10, 0.40, 0.18)
        pdf.drawString(50, current_y - 7 * line_height - 54, "AI Recommendation / Treatment")

        pdf.setFont("Helvetica", 10)
        pdf.setFillColorRGB(0, 0, 0)
        wrapped_suggestion = textwrap.wrap(suggestion, width=85)
        text_object = pdf.beginText(50, current_y - 7 * line_height - 72)
        text_object.setLeading(14)
        for line in wrapped_suggestion:
            text_object.textLine(line)
        pdf.drawText(text_object)

        image_frame_x = 50
        image_frame_y = 110
        image_frame_width = page_width - 100
        image_frame_height = 230

        pdf.setStrokeColorRGB(0.78, 0.78, 0.78)
        pdf.setLineWidth(1)
        pdf.roundRect(image_frame_x, image_frame_y, image_frame_width, image_frame_height, radius=10, stroke=1, fill=0)

        pdf.setFont("Helvetica-Bold", 12)
        pdf.setFillColorRGB(0.10, 0.40, 0.18)
        pdf.drawString(image_frame_x + 10, image_frame_y + image_frame_height + 16, "Uploaded Leaf Image")

        if report_image is not None:
            img_width, img_height = report_image.size
            max_width = image_frame_width - 30
            max_height = image_frame_height - 30
            scale = min(max_width / img_width, max_height / img_height, 1)
            display_width = img_width * scale
            display_height = img_height * scale
            display_x = image_frame_x + (image_frame_width - display_width) / 2
            display_y = image_frame_y + (image_frame_height - display_height) / 2
            pdf.drawImage(
                ImageReader(report_image),
                display_x,
                display_y,
                width=display_width,
                height=display_height,
                preserveAspectRatio=True,
                anchor="c"
            )
        else:
            pdf.setFont("Helvetica-Oblique", 10)
            pdf.setFillColorRGB(0.45, 0.45, 0.45)
            pdf.drawCentredString(image_frame_x + image_frame_width / 2, image_frame_y + image_frame_height / 2, "Image preview not available")

        pdf.setFont("Helvetica", 9)
        pdf.setFillColorRGB(0.45, 0.45, 0.45)
        pdf.drawCentredString(page_width / 2, 40, "Generated automatically by AgriScan AI")

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
    # Local development only. Render should use Gunicorn instead.
    app.run(debug=False, port=5000)