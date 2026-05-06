from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import ezdxf
import os
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "demo_key_6688")

REFERENCE_LENGTH_MM = 25.4
FOAM_THICKNESS_MM = 30
MARGIN_MM = 5
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image file")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def get_pixel_to_mm_ratio(contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 800 < area < 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            pixel_length = max(w, h)
            return REFERENCE_LENGTH_MM / pixel_length
    raise ValueError("1-inch reference coin not found in photo")

def get_product_dimensions(contours, ratio):
    max_area = 0
    product_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > 10000:
            max_area = area
            product_contour = cnt
    if product_contour is None:
        raise ValueError("Can not detect product outline")
    x, y, w_pixel, h_pixel = cv2.boundingRect(product_contour)
    return w_pixel * ratio, h_pixel * ratio

def generate_dxf(w_mm, h_mm):
    board_w = w_mm + 2 * MARGIN_MM
    board_h = h_mm + 2 * MARGIN_MM
    cutout_w = w_mm
    cutout_h = h_mm

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.new(name="CUT", dxfattribs={"color": 7})
    doc.layers.new(name="CUTOUT", dxfattribs={"color": 1})

    msp.add_lwpolyline([
        (0, 0), (board_w, 0), (board_w, board_h), (0, board_h), (0, 0)
    ], dxfattribs={"layer": "CUT"})

    offset_x = MARGIN_MM
    offset_y = MARGIN_MM
    msp.add_lwpolyline([
        (offset_x, offset_y),
        (offset_x + cutout_w, offset_y),
        (offset_x + cutout_w, offset_y + cutout_h),
        (offset_x, offset_y + cutout_h),
        (offset_x, offset_y)
    ], dxfattribs={"layer": "CUTOUT"})

    buf = BytesIO()
    doc.write(buf)
    buf.seek(0)
    return buf

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify(error="No file uploaded"), 400
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify(error="Empty file"), 400

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        _, contours = preprocess_image(save_path)
        ratio = get_pixel_to_mm_ratio(contours)
        w, h = get_product_dimensions(contours, ratio)
        dxf_file = generate_dxf(w, h)

        return send_file(
            dxf_file,
            as_attachment=True,
            download_name="foam_pack.dxf",
            mimetype="application/dxf"
        )
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
