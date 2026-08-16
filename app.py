import os
import json
import glob
import uuid

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from analyzer import analyze_image
from scoring import compute_combined_score
from image_compare import compare_product_images
from review_analyzer import (
    analyze_review_text,
    analyze_rating_pattern,
    analyze_price,
    analyze_review_velocity,
    analyze_seller_tenure,
    split_reviews,
    extract_reviews_from_csv,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REVIEW_IMAGES_DIR = os.path.join(BASE_DIR, "static", "review_images")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB (image + buyer photos)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo-products")
def demo_products():
    demo_path = os.path.join(BASE_DIR, "data", "demo_products.json")
    try:
        with open(demo_path, encoding="utf-8") as f:
            products = json.load(f)
    except FileNotFoundError:
        return jsonify([])

    # Auto-detect any buyer-uploaded review images dropped into
    # static/review_images/<product_id>/ - no JSON editing required.
    for product in products:
        folder = os.path.join(REVIEW_IMAGES_DIR, product["id"])
        buyer_images = []
        if os.path.isdir(folder):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                for path in sorted(glob.glob(os.path.join(folder, ext))):
                    rel = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
                    buyer_images.append("/" + rel)
        product["buyer_images"] = buyer_images

    return jsonify(products)


@app.route("/analyze", methods=["POST"])
def analyze():
    # ---- Listing image (required) ----
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_image(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP or BMP."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
    file.save(filepath)

    # ---- Buyer/reviewer photos (optional, multiple) ----
    buyer_temp_paths = []
    buyer_files = request.files.getlist("buyer_images")
    for bf in buyer_files:
        if bf and bf.filename and allowed_image(bf.filename):
            bname = secure_filename(bf.filename)
            bpath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{bname}")
            bf.save(bpath)
            buyer_temp_paths.append(bpath)

    try:
        features = analyze_image(filepath)
        image_match_result = compare_product_images(filepath, buyer_temp_paths) if buyer_temp_paths else None
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Image analysis failed: {exc}"}), 500
    finally:
        # No database / no persistence - remove everything right after analysis
        if os.path.exists(filepath):
            os.remove(filepath)
        for bpath in buyer_temp_paths:
            if os.path.exists(bpath):
                os.remove(bpath)

    # ---- Listing signals (all optional) ----
    price = parse_float(request.form.get("price"))
    expected_price = parse_float(request.form.get("expected_price"))
    avg_rating = parse_float(request.form.get("avg_rating"))
    num_ratings = parse_int(request.form.get("num_ratings"))
    review_text = request.form.get("review_text", "")
    seller_months_active = parse_int(request.form.get("seller_months_active"))

    review_timestamps = []
    raw_timestamps = request.form.get("review_timestamps", "")
    if raw_timestamps:
        try:
            parsed = json.loads(raw_timestamps)
            if isinstance(parsed, list):
                review_timestamps = parsed
        except (ValueError, TypeError):
            review_timestamps = []

    reviews = split_reviews(review_text)

    reviews_csv = request.files.get("reviews_csv")
    if reviews_csv and reviews_csv.filename:
        try:
            reviews += extract_reviews_from_csv(reviews_csv.stream)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Could not read reviews CSV: {exc}"}), 400

    try:
        review_result = analyze_review_text(reviews)
        rating_result = analyze_rating_pattern(avg_rating, num_ratings)
        price_result = analyze_price(price, expected_price)
        velocity_result = analyze_review_velocity(review_timestamps)
        seller_result = analyze_seller_tenure(seller_months_active)
        result = compute_combined_score(
            features,
            review_result=review_result,
            rating_result=rating_result,
            price_result=price_result,
            velocity_result=velocity_result,
            seller_result=seller_result,
            image_match_result=image_match_result,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Scoring failed: {exc}"}), 500

    result["meta"] = {
        "width": features["width"],
        "height": features["height"],
        "filename": filename,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
