# -*- coding: utf-8 -*-
"""
Palm Line Drawer — 在 ROI 上精准绘制掌纹线条

使用 Gabor 滤波器 + 自适应阈值 + 形态学处理提取掌纹线条，
然后用骨架化和轮廓检测精准绘制生命线、智慧线、感情线。
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def _gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
    """生成 Gabor 滤波器核。"""
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)


def enhance_palm_lines(roi_gray):
    """
    用多方向 Gabor 滤波器增强掌纹线条。
    
    Args:
        roi_gray: 灰度 ROI 图像
    
    Returns:
        enhanced: 增强后的掌纹图像
    """
    if roi_gray is None or roi_gray.size == 0:
        return None
    
    # 预处理：CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(roi_gray)
    
    # 高斯模糊去噪
    eq = cv2.GaussianBlur(eq, (5, 5), 1.0)
    
    # Gabor 滤波器参数（针对掌纹线条优化）
    h, w = roi_gray.shape[:2]
    ksize  = max(21, int(min(h, w) * 0.08) | 1)  # 奇数
    sigma  = ksize / 6.0
    lambd  = ksize / 2.0
    gamma  = 0.5
    psi    = 0
    
    # 多方向滤波（8 个方向）
    angles = [i * np.pi / 8 for i in range(8)]
    responses = []
    
    for theta in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(eq, cv2.CV_32F, kernel)
        responses.append(np.abs(filtered))
    
    # 取最大响应
    enhanced = np.maximum.reduce(responses)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return enhanced


def extract_palm_lines_binary(roi_gray):
    """
    从 ROI 提取二值化的掌纹线条。
    
    Returns:
        binary: 二值图像（白色=掌纹线条）
    """
    if roi_gray is None or roi_gray.size == 0:
        return None
    
    # Gabor 增强
    enhanced = enhance_palm_lines(roi_gray)
    if enhanced is None:
        return None
    
    # 自适应阈值（使用较大的块大小减少碎片）
    block_size = max(31, int(min(roi_gray.shape) * 0.12) | 1)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 8
    )
    
    # 形态学处理：先闭合断裂，再去除小噪点
    h, w = roi_gray.shape[:2]
    k_close = max(3, int(min(h, w) * 0.015))
    k_open  = max(3, int(min(h, w) * 0.01))
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    
    # 去除面积太小的连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(h * w * 0.005)  # 至少占 0.5% 面积
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0
    
    return binary


def identify_major_lines(binary, roi_shape):
    """
    从二值掌纹图像中识别三条主要线条（生命线、智慧线、感情线）。
    
    策略（基于 ROI 区域的相对位置）：
      - 感情线：上部 1/3 区域，横向为主
      - 智慧线：中部区域，横向为主
      - 生命线：左侧弧形，纵向跨度大
    
    Returns:
        dict with keys 'life', 'head', 'heart', each containing contour points
    """
    if binary is None or binary.size == 0:
        return dict(life=None, head=None, heart=None)
    
    h, w = roi_shape[:2]
    
    # 骨架化
    if hasattr(cv2, 'ximgproc'):
        skeleton = cv2.ximgproc.thinning(binary)
    else:
        skeleton = binary
    
    # 查找轮廓
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return dict(life=None, head=None, heart=None)
    
    # 过滤太短的轮廓，按长度排序
    min_length = max(w, h) * 0.15
    valid_contours = [(cv2.arcLength(c, False), c) for c in contours
                      if cv2.arcLength(c, False) > min_length]
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    valid_contours = valid_contours[:15]
    
    if not valid_contours:
        return dict(life=None, head=None, heart=None)
    
    # 为每条轮廓计算特征
    candidates = []
    for arc_len, cnt in valid_contours:
        pts = cnt.reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # 方向性：横向 vs 纵向
        aspect = x_span / (y_span + 1e-6)
        
        candidates.append(dict(
            cnt=cnt, arc_len=arc_len,
            x_center=x_center, y_center=y_center,
            x_span=x_span, y_span=y_span,
            aspect=aspect,
            y_min=y_min, y_max=y_max,
            x_min=x_min, x_max=x_max,
        ))
    
    # 分类策略
    life_best  = None
    head_best  = None
    heart_best = None
    
    life_score  = -1
    head_score  = -1
    heart_score = -1
    
    for c in candidates:
        # 感情线：上部区域（y_center < h*0.4），横向为主（aspect > 1.0）
        if c['y_center'] < h * 0.45 and c['aspect'] > 0.8 and c['x_span'] > w * 0.25:
            score = c['arc_len'] * (1.0 - c['y_center'] / h)
            if score > heart_score:
                heart_score = score
                heart_best  = c['cnt']
        
        # 智慧线：中部区域（h*0.3 < y_center < h*0.7），横向为主
        if c['y_center'] > h * 0.3 and c['y_center'] < h * 0.75 and c['aspect'] > 0.6 and c['x_span'] > w * 0.25:
            score = c['arc_len'] * (1.0 - abs(c['y_center'] / h - 0.5))
            if score > head_score:
                head_score = score
                head_best  = c['cnt']
        
        # 生命线：左侧（x_center < w*0.5），纵向跨度大（y_span > h*0.25）
        if c['x_center'] < w * 0.55 and c['y_span'] > h * 0.2:
            score = c['arc_len'] * (1.0 - c['x_center'] / w)
            if score > life_score:
                life_score = score
                life_best  = c['cnt']
    
    # 避免同一条轮廓被分配给多个线条
    assigned = set()
    result = dict(life=None, head=None, heart=None)
    
    # 按优先级分配（感情线 > 智慧线 > 生命线）
    for name, cnt, score in [
        ('heart', heart_best, heart_score),
        ('head',  head_best,  head_score),
        ('life',  life_best,  life_score),
    ]:
        if cnt is not None and id(cnt) not in assigned:
            result[name] = cnt
            assigned.add(id(cnt))
    
    return result


def draw_palm_lines_on_roi(roi_bgr, lines_dict, thickness=2):
    """
    在 ROI 图像上绘制识别出的掌纹线条。
    
    Args:
        roi_bgr: BGR ROI 图像
        lines_dict: dict with keys 'life', 'head', 'heart' (contours)
        thickness: 线条粗细
    
    Returns:
        annotated: 标注后的图像
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    canvas = roi_bgr.copy()
    
    colors = dict(
        life  = (220,  55,  55),   # 红色
        head  = ( 55, 200,  75),   # 绿色
        heart = ( 60, 120, 220),   # 蓝色
    )
    
    for name, cnt in lines_dict.items():
        if cnt is not None and len(cnt) > 0:
            cv2.drawContours(canvas, [cnt], -1, colors[name], thickness, cv2.LINE_AA)
    
    return canvas


def extract_features_from_lines(lines_dict, roi_shape):
    """
    从识别出的线条提取几何特征（长度、曲率、角度等）。
    
    Returns:
        dict with feature names as keys
    """
    features = {}
    h, w = roi_shape[:2]
    
    for name in ['life', 'head', 'heart']:
        cnt = lines_dict.get(name)
        
        if cnt is None or len(cnt) < 2:
            features[f'{name}_length']    = 0.0
            features[f'{name}_curvature'] = 0.0
            features[f'{name}_angle']     = 0.0
        else:
            pts = cnt.reshape(-1, 2).astype(np.float64)
            
            # 长度
            arc_len = cv2.arcLength(cnt, False)
            features[f'{name}_length'] = arc_len
            
            # 曲率（弯曲度）
            straight = np.linalg.norm(pts[-1] - pts[0])
            features[f'{name}_curvature'] = arc_len / straight if straight > 1 else 1.0
            
            # 角度
            delta = pts[-1] - pts[0]
            angle = np.degrees(np.arctan2(delta[1], delta[0]))
            features[f'{name}_angle'] = float(angle)
    
    # 交叉点（简化：检测线条间的最近距离）
    def _approx_intersections(cnt_a, cnt_b, tol=5):
        if cnt_a is None or cnt_b is None:
            return 0
        pts_a = cnt_a.reshape(-1, 2)
        pts_b = cnt_b.reshape(-1, 2)
        count = 0
        for pa in pts_a[::5]:
            dists = np.linalg.norm(pts_b.astype(float) - pa.astype(float), axis=1)
            if dists.min() < tol:
                count += 1
        return min(count, 5)
    
    features['life_head_intersection']  = _approx_intersections(lines_dict.get('life'), lines_dict.get('head'))
    features['life_heart_intersection'] = _approx_intersections(lines_dict.get('life'), lines_dict.get('heart'))
    features['head_heart_intersection'] = _approx_intersections(lines_dict.get('head'), lines_dict.get('heart'))
    features['line_coverage'] = 0.0
    
    return features


def process_palm_lines(roi_bgr):
    """
    完整的掌纹线条处理流程。
    
    Args:
        roi_bgr: BGR ROI 图像
    
    Returns:
        dict with keys:
          'annotated'  — 标注了掌纹线条的图像
          'binary'     — 二值掌纹图像
          'lines'      — dict of contours (life, head, heart)
          'features'   — dict of geometric features
          'error'      — error message or None
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return dict(annotated=None, binary=None, lines=None, features=None,
                    error="ROI 图像为空")
    
    # 转灰度
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
    
    # 提取二值掌纹
    binary = extract_palm_lines_binary(gray)
    if binary is None:
        return dict(annotated=roi_bgr, binary=None, lines=None, features=None,
                    error="掌纹增强失败")
    
    # 识别主要线条
    lines = identify_major_lines(binary, roi_bgr.shape)
    
    # 绘制线条
    annotated = draw_palm_lines_on_roi(roi_bgr, lines, thickness=2)
    
    # 提取特征
    features = extract_features_from_lines(lines, roi_bgr.shape)
    
    return dict(annotated=annotated, binary=binary, lines=lines,
                features=features, error=None)
