# -*- coding: utf-8 -*-
"""
Palm Line Analysis — Gradio Web Interface

集成 Huterox/palm_recongnition 的 ROI 提取流程：
  1. 手部关键点检测（OpenCV DNN + MediaPipe ONNX）
  2. 图像旋转矫正
  3. GrabCut 手部分割
  4. PSO 算法精确定位掌心 ROI
  5. Gabor 滤波器 + 骨架化精准绘制掌纹线条
"""

import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image

from utils.palm_roi_extractor import extract_palm_roi
from utils.palm_line_drawer   import process_palm_lines


# ── 分类逻辑 ──────────────────────────────────────────────────────────────────

def _classify(features: dict) -> dict:
    lengths = {k: features.get(f"{k}_length", 0) for k in ("life", "head", "heart")}
    total   = sum(lengths.values())
    if total == 0:
        dominant, conf = "Unknown", 0.0
    else:
        dominant = max(lengths, key=lengths.get)
        conf     = round(lengths[dominant] / total, 3)

    avg_curv = sum(features.get(f"{k}_curvature", 0) for k in ("life", "head", "heart")) / 3
    if avg_curv > 1.3:
        palm_type = "Curved / Expressive"
    elif avg_curv > 1.1:
        palm_type = "Balanced"
    else:
        palm_type = "Straight / Practical"

    head_angle    = abs(features.get("head_angle", 0))
    intersections = features.get("life_head_intersection", 0)
    career, cs_conf = ("Yes", 0.70) if (head_angle > 10 and intersections > 0) else ("No", 0.60)

    return dict(dominant_line=dominant, confidence=conf, palm_type=palm_type,
                career_shift_indicator=career, career_shift_confidence=cs_conf)


def _fmt_features(features: dict) -> str:
    lines = [
        "## 掌纹特征\n",
        "### 线条长度（像素）",
        f"- 🔴 **生命线**: {features['life_length']:.1f}",
        f"- 🟢 **智慧线**: {features['head_length']:.1f}",
        f"- 🔵 **感情线**: {features['heart_length']:.1f}\n",
        "### 曲率",
        f"- 生命线: {features['life_curvature']:.3f}",
        f"- 智慧线: {features['head_curvature']:.3f}",
        f"- 感情线: {features['heart_curvature']:.3f}\n",
        "### 角度（°）",
        f"- 生命线: {features['life_angle']:.1f}°",
        f"- 智慧线: {features['head_angle']:.1f}°",
        f"- 感情线: {features['heart_angle']:.1f}°\n",
        "### 交叉点",
        f"- 生命线 × 智慧线: {features['life_head_intersection']}",
        f"- 生命线 × 感情线: {features['life_heart_intersection']}",
        f"- 智慧线 × 感情线: {features['head_heart_intersection']}",
    ]
    return "\n".join(lines)


def _fmt_classification(cls: dict) -> str:
    lines = [
        "## 掌纹分析\n",
        "### 主导线条",
        f"**{cls['dominant_line']} Line** — 置信度 {cls['confidence']:.1%}\n",
        "### 手型",
        f"**{cls['palm_type']}**\n",
        "### 事业转变指标",
        f"**{cls['career_shift_indicator']}** — 置信度 {cls['career_shift_confidence']:.1%}",
    ]
    return "\n".join(lines)


# ── 主处理函数 ────────────────────────────────────────────────────────────────

def process_palm_image(image, show_roi: bool, show_binary: bool, thickness: int):
    """
    完整掌纹分析流程。
    
    Returns:
        (result_image, roi_image, features_text, classification_text)
    """
    if image is None:
        return None, None, "请先上传手掌图片。", ""

    # 统一转为 BGR numpy
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = np.asarray(image)
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # ── Step 1-5: ROI 提取（Huterox/palm_recongnition 流程）─────────────────
    roi_result = extract_palm_roi(img_bgr)

    if roi_result['error'] is not None:
        err_msg = f"## ⚠️ ROI 提取失败\n\n{roi_result['error']}\n\n**建议**：\n- 确保手掌正面朝向镜头\n- 光线充足，背景简洁\n- 手指展开，手掌平放"
        # 返回原图
        img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb_out, None, err_msg, ""

    roi_square = roi_result['roi_square']   # BGR
    draw_img   = roi_result['draw_img']     # BGR，带 ROI 标注

    # ── Step 6: 掌纹线条提取和绘制 ───────────────────────────────────────────
    line_result = process_palm_lines(roi_square)

    # 构建输出图像
    if line_result['annotated'] is not None:
        # 将标注后的 ROI 转为 RGB 输出
        annotated_rgb = cv2.cvtColor(line_result['annotated'], cv2.COLOR_BGR2RGB)
        roi_out = annotated_rgb
    else:
        roi_out = cv2.cvtColor(roi_square, cv2.COLOR_BGR2RGB) if roi_square is not None else None

    # 主图：显示 ROI 标注框（可选）
    if show_roi:
        main_out = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    else:
        main_out = cv2.cvtColor(roi_result['rotated_img'], cv2.COLOR_BGR2RGB)

    # 特征和分类
    if line_result['features'] is not None:
        features = line_result['features']
        cls      = _classify(features)
        feat_txt = _fmt_features(features)
        cls_txt  = _fmt_classification(cls)
    else:
        feat_txt = f"## ⚠️ 掌纹提取失败\n\n{line_result.get('error', '未知错误')}"
        cls_txt  = ""

    return main_out, roi_out, feat_txt, cls_txt


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="掌纹精准检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖐️ 掌纹精准检测与分析")
    gr.Markdown(
        "基于 [Huterox/palm_recongnition](https://github.com/Huterox/palm_recongnition) 的 ROI 提取流程：\n\n"
        "**关键点检测 → 旋转矫正 → 手部分割 → PSO 掌心定位 → Gabor 掌纹增强**\n\n"
        "| 颜色 | 掌纹线 |\n|:---:|:---|\n"
        "| 🔴 | 生命线 Life Line |\n"
        "| 🟢 | 智慧线 Head Line |\n"
        "| 🔵 | 感情线 Heart Line |"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传手掌图片", type="pil")
            with gr.Accordion("选项", open=False):
                show_roi    = gr.Checkbox(label="主图显示 ROI 定位框", value=True)
                show_binary = gr.Checkbox(label="显示二值掌纹图", value=False)
                thickness   = gr.Slider(1, 5, value=2, step=1, label="线条粗细")
            analyze_btn = gr.Button("🔍 分析掌纹", variant="primary", size="lg")

        with gr.Column(scale=1):
            main_out = gr.Image(label="ROI 定位结果", type="numpy")

    with gr.Row():
        roi_out = gr.Image(label="掌纹线条（ROI 区域）", type="numpy")

    with gr.Row():
        features_out       = gr.Markdown()
        classification_out = gr.Markdown()

    analyze_btn.click(
        fn=process_palm_image,
        inputs=[input_image, show_roi, show_binary, thickness],
        outputs=[main_out, roi_out, features_out, classification_out],
    )

    gr.Markdown("""
---
**使用说明**：上传手掌正面照片（光线充足、手指展开），点击「分析掌纹」。

**流程说明**：
1. OpenCV DNN 检测 21 个手部关键点
2. 根据腕部→中指向量旋转矫正图像
3. GrabCut 分割手部区域
4. PSO 算法精确定位掌心圆心和最大内接圆
5. Gabor 滤波器增强掌纹，骨架化后识别三条主线

> 免责声明：本系统仅供技术演示，掌纹分析结果不具备科学依据。
""")

if __name__ == "__main__":
    demo.launch(share=False)
