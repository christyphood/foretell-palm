"""
Palm Line Segmentation API
使用 Palm-Astro 的 U-Net 模型进行掌纹语义分割
返回3条主线的轮廓点数据供前端绘制
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Palm-Astro-Application'))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
CORS(app)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def serve_index():
    """与 /analyze 同源提供前端，避免只打开根路径时出现 404。"""
    return send_from_directory(_REPO_ROOT, "index.html")


# 模型加载
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Palm-Astro-Application', 'results', 'best_model.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
try:
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
except Exception as e:
    print(f"Model load error: {e}")


def get_preprocessing():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def extract_line_contour(mask, class_id):
    """从分割掩码中提取指定类别的线条轮廓点（归一化坐标）"""
    binary = (mask == class_id).astype(np.uint8) * 255
    if binary.max() == 0:
        return None

    h, w = binary.shape

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # 取最大轮廓
    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)

    if len(points) < 5:
        return None

    # 骨架化获取中心线
    kernel = np.ones((3, 3), np.uint8)
    # 使用形态学操作获取中心线
    thinned = binary.copy()
    for _ in range(5):
        eroded = cv2.erode(thinned, kernel)
        if eroded.max() == 0:
            break
        thinned = eroded

    # 如果骨架化失败，用轮廓中心线
    # 按主方向排序取中值
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None

    # 判断主方向
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()

    line_points = []
    if x_range >= y_range:
        # 水平方向线 - 按x采样
        step = max(1, x_range // 25)
        for x in range(xs.min(), xs.max() + 1, step):
            col_ys = ys[xs == x]
            if len(col_ys) > 0:
                mid_y = col_ys.mean()
                line_points.append([float(x) / w, float(mid_y) / h])
    else:
        # 垂直方向线 - 按y采样
        step = max(1, y_range // 25)
        for y in range(ys.min(), ys.max() + 1, step):
            row_xs = xs[ys == y]
            if len(row_xs) > 0:
                mid_x = row_xs.mean()
                line_points.append([float(mid_x) / w, float(y) / h])

    if len(line_points) < 3:
        return None

    # 平滑
    pts = np.array(line_points)
    window = 5
    if len(pts) > window:
        smoothed = []
        for i in range(len(pts)):
            start = max(0, i - window // 2)
            end = min(len(pts), i + window // 2 + 1)
            smoothed.append(pts[start:end].mean(axis=0).tolist())
        line_points = smoothed

    return line_points


@app.route('/analyze', methods=['POST'])
def analyze():
    """接收图片，返回3条掌纹线的坐标点"""
    try:
        data = request.json
        image_b64 = data.get('image', '')

        # 解码base64图片
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img_pil)

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # 预处理
        preprocessing = get_preprocessing()
        preprocessed = preprocessing(image=img_np)
        image_tensor = preprocessed['image'].unsqueeze(0).to(DEVICE)

        # 推理
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # 放大到原图尺寸
        mask = cv2.resize(pred_mask.astype(np.uint8),
                         (img_np.shape[1], img_np.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

        # 提取3条线的轮廓点
        result = {
            'life_line': extract_line_contour(mask, 1),    # 生命线
            'head_line': extract_line_contour(mask, 2),    # 头脑线
            'heart_line': extract_line_contour(mask, 3),   # 感情线
            'success': True
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
