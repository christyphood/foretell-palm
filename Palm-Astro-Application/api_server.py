# -*- coding: utf-8 -*-
"""
Palm Recognition API Server
基于 Huterox/palm_recongnition 的掌纹识别 Flask API

为 index.html 前端提供 /analyze 接口：
  1. 接收 base64 编码的手掌图片
  2. 执行 ROI 提取（关键点检测 → 旋转矫正 → 手部分割 → PSO 掌心定位）
  3. 执行掌纹线条提取（Gabor 滤波 → 骨架化 → 三线识别）
  4. 返回归一化的掌纹线条坐标和特征分析
"""

import base64
import io
import os
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from utils.palm_roi_extractor import extract_palm_roi
from utils.palm_line_drawer import process_palm_lines

# palm 仓库根目录（与 Palm-Astro-Application 同级，含 index.html）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)
CORS(app)


@app.route("/")
def serve_index():
    """与 API 同源提供前端，避免 file:// / 跨端口 / HTTPS 混合内容导致无法请求 /analyze。"""
    return send_from_directory(_REPO_ROOT, "index.html")


def decode_base64_image(b64_string):
    """解码 base64 图片为 BGR numpy 数组。"""
    # 去掉 data:image/...;base64, 前缀
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


def contour_to_normalized_points(contour, shape):
    """将轮廓点转换为归一化坐标列表 [[x, y], ...]。"""
    if contour is None or len(contour) == 0:
        return None
    h, w = shape[:2]
    pts = contour.reshape(-1, 2)
    # 均匀采样，最多返回 30 个点以减少数据量
    if len(pts) > 30:
        indices = np.linspace(0, len(pts) - 1, 30, dtype=int)
        pts = pts[indices]
    normalized = [[float(p[0]) / w, float(p[1]) / h] for p in pts]
    return normalized


def image_to_base64(img_bgr):
    """将 BGR 图像编码为 base64 data URL。"""
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def _classify(features: dict) -> dict:
    """基于特征进行掌纹分类分析。"""
    lengths = {k: features.get(f"{k}_length", 0) for k in ("life", "head", "heart")}
    total = sum(lengths.values())
    if total == 0:
        dominant, conf = "unknown", 0.0
    else:
        dominant = max(lengths, key=lengths.get)
        conf = round(lengths[dominant] / total, 3)

    avg_curv = sum(features.get(f"{k}_curvature", 0) for k in ("life", "head", "heart")) / 3
    if avg_curv > 1.3:
        palm_type = "curved_expressive"
    elif avg_curv > 1.1:
        palm_type = "balanced"
    else:
        palm_type = "straight_practical"

    return dict(
        dominant_line=dominant,
        confidence=conf,
        palm_type=palm_type,
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    掌纹分析 API。

    请求体 JSON:
      { "image": "data:image/jpeg;base64,..." }

    响应 JSON:
      {
        "success": true/false,
        "heart_line": [[x, y], ...] | null,   (归一化坐标 0~1，相对于原图)
        "head_line": [[x, y], ...] | null,
        "life_line": [[x, y], ...] | null,
        "features": { ... },
        "classification": { ... },
        "roi_center": [x, y],   (归一化)
        "roi_radius": float,    (归一化，相对于图像短边)
        "roi_image": "data:image/jpeg;base64,...",  (标注了线条的ROI图)
        "annotated_image": "data:image/jpeg;base64,...",  (标注了ROI框的全图)
        "error": null | "错误信息"
      }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "缺少 image 字段"}), 400

        # 解码图片
        img_bgr = decode_base64_image(data['image'])
        h_orig, w_orig = img_bgr.shape[:2]

        # Step 1-5: ROI 提取（Huterox/palm_recongnition 流程）
        roi_result = extract_palm_roi(img_bgr)

        if roi_result['error'] is not None:
            return jsonify({
                "success": False,
                "error": roi_result['error'],
                "heart_line": None,
                "head_line": None,
                "life_line": None,
                "features": None,
                "classification": None,
                "roi_center": None,
                "roi_radius": None,
                "roi_image": None,
                "annotated_image": None,
            })

        roi_square = roi_result['roi_square']

        # Step 6: 掌纹线条提取
        line_result = process_palm_lines(roi_square)

        # 构建响应
        response = {
            "success": True,
            "error": None,
            "heart_line": None,
            "head_line": None,
            "life_line": None,
            "features": None,
            "classification": None,
            "roi_center": None,
            "roi_radius": None,
            "roi_image": None,
            "annotated_image": None,
        }

        # ROI 信息（归一化）
        proc_h, proc_w = roi_result['rotated_img'].shape[:2]
        if roi_result['center'] is not None:
            cx, cy = roi_result['center']
            response['roi_center'] = [float(cx) / proc_w, float(cy) / proc_h]
            response['roi_radius'] = float(roi_result['radius']) / min(proc_w, proc_h)

        # 返回标注了 ROI 框的全图
        if roi_result['draw_img'] is not None:
            response['annotated_image'] = image_to_base64(roi_result['draw_img'])

        # 掌纹线条坐标（归一化到旋转后的全图坐标系）
        if line_result['lines'] is not None:
            roi_h, roi_w = roi_square.shape[:2]
            lines = line_result['lines']

            # ROI 正方形在旋转后图像中的位置
            # ROI 是以 center 为中心、边长 = radius * sqrt(2) 的正方形
            radius = roi_result['radius']
            cx, cy = roi_result['center']
            side = int(radius * np.sqrt(2))
            roi_x1 = max(0, int(cx) - side // 2)
            roi_y1 = max(0, int(cy) - side // 2)

            # 将 ROI 内的轮廓坐标映射到旋转后全图的归一化坐标
            for line_key, resp_key in [('heart', 'heart_line'), ('head', 'head_line'), ('life', 'life_line')]:
                cnt = lines.get(line_key)
                if cnt is not None and len(cnt) > 0:
                    pts = cnt.reshape(-1, 2)
                    if len(pts) > 30:
                        indices = np.linspace(0, len(pts) - 1, 30, dtype=int)
                        pts = pts[indices]
                    # 映射: ROI局部坐标 → 全图坐标 → 归一化
                    normalized = [
                        [float(roi_x1 + p[0]) / proc_w,
                         float(roi_y1 + p[1]) / proc_h]
                        for p in pts
                    ]
                    response[resp_key] = normalized

        # 返回标注了线条的 ROI 图片
        if line_result['annotated'] is not None:
            response['roi_image'] = image_to_base64(line_result['annotated'])

        # 特征和分类
        if line_result['features'] is not None:
            features = line_result['features']
            response['features'] = {
                "life_length": round(features.get('life_length', 0), 1),
                "head_length": round(features.get('head_length', 0), 1),
                "heart_length": round(features.get('heart_length', 0), 1),
                "life_curvature": round(features.get('life_curvature', 0), 3),
                "head_curvature": round(features.get('head_curvature', 0), 3),
                "heart_curvature": round(features.get('heart_curvature', 0), 3),
                "life_angle": round(features.get('life_angle', 0), 1),
                "head_angle": round(features.get('head_angle', 0), 1),
                "heart_angle": round(features.get('heart_angle', 0), 1),
            }
            response['classification'] = _classify(features)

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"服务器内部错误: {str(e)}",
            "heart_line": None,
            "head_line": None,
            "life_line": None,
            "features": None,
            "classification": None,
            "roi_center": None,
            "roi_radius": None,
            "roi_image": None,
            "annotated_image": None,
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口。"""
    return jsonify({"status": "ok", "service": "palm-recognition-api"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5050"))
    print("=" * 60)
    print("  掌纹识别 API 服务器")
    print("  基于 Huterox/palm_recongnition")
    print(f"  端口: {port} (Render 等平台通过环境变量 PORT 注入)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
