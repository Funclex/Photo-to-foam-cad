from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import ezdxf
import os
from io import BytesIO, StringIO
import base64

app = Flask(__name__)
app.secret_key = "demo_key_6688"

REFERENCE_LENGTH_MM = 25.4
FOAM_THICKNESS_MM = 30
MARGIN_MM = 5
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 处理图片 + 画框 + 返回base64给前端
def process_image_and_get_box(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找硬币(圆形)
    coin_box = None
    product_box = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        ratio = w / h

        # 硬币：接近圆形
        if 0.75 < ratio < 1.25 and 500 < area < 20000:
            coin_box = (x,y,w,h)

    # 找最大物体 = 产品
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            x,y,w,h = cv2.boundingRect(cnt)
            product_box = (x,y,w,h)

    # 画红框硬币
    if coin_box:
        x,y,w,h = coin_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img,"Coin",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # 画绿框产品
    if product_box:
        x,y,w,h = product_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img,"Product",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # 计算比例
    if not coin_box:
        raise ValueError("Coin not detected")
    if not product_box:
        raise ValueError("Product not detected")

    cx,cy,cw,ch = coin_box
    px,py,pw,ph = product_box

    pixel_per_mm = max(cw,ch) / REFERENCE_LENGTH_MM
    real_w = pw / pixel_per_mm
    real_h = ph / pixel_per_mm

    # 转base64传回前端
    _, buf = cv2.imencode(".jpg", img)
    b64_img = base64.b64encode(buf).decode()

    return b64_img, real_w, real_h

# 生成DXF
def generate_dxf(w_mm, h_mm):
    board_w = w_mm + 2 * MARGIN_MM
    board_h = h_mm + 2 * MARGIN_MM
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.new("CUT", dxfattribs={"color":7})
    doc.layers.new("CUTOUT", dxfattribs={"color":1})

    msp.add_lwpolyline([(0,0),(board_w,0),(board_w,board_h),(0,board_h),(0,0)], dxfattribs={"layer":"CUT"})
    ox = MARGIN_MM
    oy = MARGIN_MM
    msp.add_lwpolyline([(ox,oy),(ox+w_mm,oy),(ox+w_mm,oy+h_mm),(ox,oy+h_mm),(ox,oy)], dxfattribs={"layer":"CUTOUT"})

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
        return jsonify({"ok":False, "msg":str(e)})

@app.route("/download_dxf", methods=["POST"])
def download_dxf():
    w = float(request.form["w"])
    h = float(request.form["h"])
    dxf_buf = generate_dxf(w, h)
    return send_file(dxf_buf, as_attachment=True, download_name="foam_pack.dxf")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
