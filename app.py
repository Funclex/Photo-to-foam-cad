from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import ezdxf
import os
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = "demo_key_6688"

REFERENCE_LENGTH_MM = 25.4
FOAM_THICKNESS_MM = 30
MARGIN_MM = 5
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image_and_get_box(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image")

    # 直接用灰度+二值化，降低所有门槛
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_box = None
    product_box = None

    # 找硬币：宽高比接近1，面积适中
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            if 0.7 < ratio < 1.3:
                coin_box = (x, y, w, h)
                break

    # 找最大的轮廓，直接当产品
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            x, y, w, h = cv2.boundingRect(cnt)
            product_box = (x, y, w, h)

    if not coin_box:
        raise ValueError("Coin not detected")
    if not product_box:
        raise ValueError("Product not detected")

    # 画红框硬币、绿框产品
    x, y, w, h = coin_box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, "Coin", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    x, y, w, h = product_box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, "Product", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 计算实际尺寸
    cx, cy, cw, ch = coin_box
    pixel_per_mm = max(cw, ch) / REFERENCE_LENGTH_MM
    real_w = w / pixel_per_mm
    real_h = h / pixel_per_mm

    # 转base64传回前端
    _, buf = cv2.imencode(".jpg", img)
    b64_img = base64.b64encode(buf).decode()

    return b64_img, real_w, real_h

def generate_dxf(w_mm, h_mm):
    board_w = w_mm + 2 * MARGIN_MM
    board_h = h_mm + 2 * MARGIN_MM
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.new("CUT", dxfattribs={"color": 7})
    doc.layers.new("CUTOUT", dxfattribs={"color": 1})

    msp.add_lwpolyline([(0,0), (board_w,0), (board_w,board_h), (0,board_h), (0,0)], dxfattribs={"layer": "CUT"})
    ox, oy = MARGIN_MM, MARGIN_MM
    msp.add_lwpolyline([(ox,oy), (ox+w_mm,oy), (ox+w_mm,oy+h_mm), (ox,oy+h_mm), (ox,oy)], dxfattribs={"layer": "CUTOUT"})

    buf = BytesIO()
    doc.write(buf)
    buf.seek(0)
    return buf

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preview", methods=["POST"])
def preview():
    file = request.files["file"]
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        b64_img, w, h = process_image_and_get_box(save_path)
        return jsonify({
            "ok": True,
            "img": b64_img,
            "width": round(w,2),
            "height": round(h,2)
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/download_dxf", methods=["POST"])
def download_dxf():
    w = float(request.form["w"])
    h = float(request.form["h"])
    dxf_buf = generate_dxf(w, h)
    return send_file(dxf_buf, as_attachment=True, download_name="foam_pack.dxf")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
