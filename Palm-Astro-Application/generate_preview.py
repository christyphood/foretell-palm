#!/usr/bin/env python3
"""生成 HTML 预览页面"""
import cv2
import base64
import os

def img_to_b64(path, max_w=400):
    img = cv2.imread(path)
    if img is None:
        return ''
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

orig_b64   = img_to_b64('data/images/20251128_085131.jpg')
roi_b64    = img_to_b64('/tmp/palm_roi_annotated.jpg')
square_b64 = img_to_b64('/tmp/palm_roi_square.jpg', 300)
lines_b64  = img_to_b64('/tmp/palm_lines_annotated.jpg', 300)
binary_b64 = img_to_b64('/tmp/palm_lines_binary.jpg', 300)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>掌纹精准检测 - 预览</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
h1 {{ text-align: center; color: #00d4ff; }}
h2 {{ color: #ffd700; border-bottom: 1px solid #333; padding-bottom: 8px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
.card {{ background: #16213e; border-radius: 12px; padding: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
.card img {{ width: 100%; border-radius: 8px; }}
.card h3 {{ margin: 10px 0 5px; color: #00d4ff; font-size: 14px; }}
.card p {{ font-size: 12px; color: #aaa; margin: 4px 0; }}
.pipeline {{ background: #0f3460; border-radius: 12px; padding: 20px; margin: 20px 0; }}
.pipeline ol {{ font-size: 14px; line-height: 2; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
.badge-red {{ background: rgba(220,55,55,0.3); color: #ff6b6b; }}
.badge-green {{ background: rgba(55,200,75,0.3); color: #69db7c; }}
.badge-blue {{ background: rgba(60,120,220,0.3); color: #74c0fc; }}
.features {{ background: #16213e; border-radius: 12px; padding: 20px; margin: 20px 0; }}
.features table {{ width: 100%; border-collapse: collapse; }}
.features td, .features th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; font-size: 13px; }}
.features th {{ color: #ffd700; }}
</style>
</head>
<body>
<div class="container">
<h1>&#x1F590;&#xFE0F; 掌纹精准检测与分析 - 预览结果</h1>

<div class="pipeline">
<h2>处理流程 (基于 Huterox/palm_recongnition)</h2>
<ol>
<li><strong>手部关键点检测</strong> - MediaPipe ONNX 模型 (21 keypoints)</li>
<li><strong>图像旋转矫正</strong> - 腕部到中指MCP向量对齐Y轴</li>
<li><strong>手部分割</strong> - GrabCut 前景提取</li>
<li><strong>PSO 掌心定位</strong> - 粒子群优化算法寻找最大内接圆</li>
<li><strong>Gabor 掌纹增强</strong> - 多方向滤波 + 骨架化</li>
<li><strong>线条识别</strong> - 轮廓分析 + 位置分类</li>
</ol>
</div>

<h2>检测结果</h2>
<div class="grid">
<div class="card">
<h3>&#x1F4F7; 原始输入 (4000x3000)</h3>
<img src="data:image/jpeg;base64,{orig_b64}" />
<p>data/images/20251128_085131.jpg</p>
</div>
<div class="card">
<h3>&#x1F3AF; ROI 定位结果</h3>
<img src="data:image/jpeg;base64,{roi_b64}" />
<p>绿色圆 = 最大内接圆 (PSO优化)<br>红色矩形 = 内切正方形 ROI<br>掌心坐标: (309, 420), 半径: 225px</p>
</div>
<div class="card">
<h3>&#x2702;&#xFE0F; 裁剪的 ROI 区域</h3>
<img src="data:image/jpeg;base64,{square_b64}" />
<p>正方形 ROI 裁剪区域</p>
</div>
</div>

<div class="grid">
<div class="card">
<h3>&#x1F50D; 掌纹线条绘制</h3>
<img src="data:image/jpeg;base64,{lines_b64}" />
<p>
<span class="badge badge-red">&#x1F534; 生命线</span>
<span class="badge badge-green">&#x1F7E2; 智慧线</span>
<span class="badge badge-blue">&#x1F535; 感情线</span>
</p>
</div>
<div class="card">
<h3>&#x26AB; 二值掌纹图</h3>
<img src="data:image/jpeg;base64,{binary_b64}" />
<p>Gabor 滤波 + 自适应阈值 + 形态学处理</p>
</div>
</div>

<div class="features">
<h2>提取的特征数据</h2>
<table>
<tr><th>特征</th><th>生命线 &#x1F534;</th><th>智慧线 &#x1F7E2;</th><th>感情线 &#x1F535;</th></tr>
<tr><td>长度 (px)</td><td>-</td><td>-</td><td>1687.1</td></tr>
<tr><td>曲率</td><td>-</td><td>-</td><td>1.000</td></tr>
<tr><td>角度</td><td>-</td><td>-</td><td>-</td></tr>
</table>
<p style="color:#ffd700; margin-top:12px;">&#x26A0;&#xFE0F; 当前版本已成功提取感情线，生命线和智慧线的分类阈值仍在优化中。</p>
</div>

<div class="pipeline">
<h2>技术说明</h2>
<ul style="font-size:13px; line-height:1.8;">
<li>&#x2705; ROI 提取流程完整移植自 <a href="https://github.com/Huterox/palm_recongnition" style="color:#00d4ff;">Huterox/palm_recongnition</a></li>
<li>&#x2705; 关键点检测使用 MediaPipe hand_landmark ONNX 模型 (onnxruntime)</li>
<li>&#x2705; PSO 算法精确定位掌心 (5粒子x30迭代)</li>
<li>&#x2705; Gabor 多方向滤波器增强掌纹纹理</li>
<li>&#x2705; 骨架化 + 轮廓分析识别主要线条</li>
<li>&#x23F3; 线条分类准确率待进一步优化</li>
</ul>
</div>
</div>
</body>
</html>'''

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Done: {out_path}')
