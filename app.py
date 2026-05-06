from flask import Flask, request, render_template_string, jsonify
import cv2
import numpy as np
import ezdxf
import os
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = "demo_key_6688"

# 配置参数（为你的照片调好了）
REFERENCE_LENGTH_MM = 25.4
FOAM_THICKNESS_MM = 30
MARGIN_MM = 5
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image_and_get_box(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image")

    # 专门适配你这张照片的识别逻辑
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 用反色二值化，强制把浅色背景和深色物体分开
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_box = None
    product_box = None

    # 1. 找硬币：放宽所有限制
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            if 0.6 < ratio < 1.4:
                coin_box = (x, y, w, h)
                break

    # 2. 找产品：直接取图片里最大的轮廓
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

    # 3. 画红框（硬币）和绿框（产品）
    x, y, w, h = coin_box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, "Coin", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    x, y, w, h = product_box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, "Product", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. 计算实际尺寸
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

# 关键：前端直接写在代码里，不用templates文件夹
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photo → Foam CAD</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen p-4">
    <div class="max-w-md mx-auto bg-white rounded-xl shadow p-6">
        <h1 class="text-xl font-bold text-center mb-4">Photo → Foam CAD</h1>
        <p class="text-sm text-gray-500 text-center mb-4">Upload photo with 1-inch coin</p>

        <input type="file" id="fileInput" accept="image/*" class="hidden">
        <label for="fileInput" class="block bg-blue-600 text-white text-center py-3 rounded-lg cursor-pointer">
            Upload Photo
        </label>

        <div class="mt-4 hidden" id="resultBox">
            <img id="detectImg" class="w-full border rounded-lg">
            <div class="text-center mt-2 text-sm">
                <span class="text-red-600">● Coin</span>
                <span class="text-green-600 ml-3">● Product</span>
            </div>
            <div class="mt-2 text-sm text-gray-700 text-center">
                Size: <span id="sizeText">-</span>
            </div>
        </div>

        <button id="dxfBtn" class="mt-4 w-full bg-green-600 text-white py-3 rounded-lg hidden">
            Download DXF CAD
        </button>

        <div id="status" class="mt-3 text-sm text-center"></div>
    </div>

    <script>
        let finalW = 0;
        let finalH = 0;

        const fileInput = document.getElementById('fileInput');
        const resultBox = document.getElementById('resultBox');
        const detectImg = document.getElementById('detectImg');
        const sizeText = document.getElementById('sizeText');
        const dxfBtn = document.getElementById('dxfBtn');
        const status = document.getElementById('status');

        fileInput.addEventListener('change', async e => {
            const f = e.target.files[0];
            if (!f) return;
            status.innerText = "Detecting coin & product...";
            resultBox.classList.add('hidden');
            dxfBtn.classList.add('hidden');

            const form = new FormData();
            form.append('file', f);

            let res = await fetch('/preview', { method: 'POST', body: form });
            let data = await res.json();

            if (!data.ok) {
                status.innerText = "❌ " + data.msg;
                return;
            }

            detectImg.src = "data:image/jpeg;base64," + data.img;
            sizeText.innerText = data.width + " mm × " + data.height + " mm";
            finalW = data.width;
            finalH = data.height;

            resultBox.classList.remove('hidden');
            dxfBtn.classList.remove('hidden');
            status.innerText = "✅ Detected successfully";
        });

        dxfBtn.addEventListener('click', async () => {
            status.innerText = "Generating DXF...";
            const form = new FormData();
            form.append("w", finalW);
            form.append("h", finalH);

            let res = await fetch('/download_dxf', { method: 'POST', body: form });
            let blob = await res.blob();
            let url = URL.createObjectURL(blob);
            let a = document.createElement("a");
            a.href = url;
            a.download = "foam_pack.dxf";
            a.click();
            URL.revokeObjectURL(url);
            status.innerText = "✅ DXF downloaded";
        });
    </script>
</body>
</html>
    """)

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
