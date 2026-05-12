# -*- coding: utf-8 -*-
"""
Palm Line Drawer — 在 ROI 上精准绘制掌纹线条

使用 Gabor 滤波器 + 自适应阈值 + 形态学处理提取掌纹线条，
骨架化后检索候选轮廓，按几何位置与「脊线响应强度」联合打分，
最后将轮廓顶点吸附到 Gabor 响应局部极大，使三条主线更贴皮肤纹理。
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def enhance_palm_lines(roi_gray):
    """
    用多方向 Gabor 滤波器增强掌纹线条。

    细线保留：双边滤波保边去噪 + CLAHE + 略轻模糊 + 12 方向 Gabor，
    使脊线响应更贴真实皮肤纹理走向。
    """
    if roi_gray is None or roi_gray.size == 0:
        return None

    # 双边滤波：压噪同时尽量不切细掌纹
    smooth = cv2.bilateralFilter(roi_gray, d=5, sigmaColor=40, sigmaSpace=40)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(smooth)

    # 轻模糊，避免抹掉细纹（原 5x5 易糊线）
    eq = cv2.GaussianBlur(eq, (3, 3), 0.6)

    h, w = roi_gray.shape[:2]
    ksize = max(21, int(min(h, w) * 0.07) | 1)
    sigma = ksize / 6.0
    lambd = max(ksize / 2.2, 5.0)
    gamma = 0.45
    psi = 0

    # 12 方向，更密采样纹理走向
    angles = [i * np.pi / 12 for i in range(12)]
    responses = []
    for theta in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(eq, cv2.CV_32F, kernel)
        responses.append(np.abs(filtered))

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
    enhanced = enhance_palm_lines(roi_gray)
    if enhanced is None:
        return None
    return _binarize_enhanced(enhanced, roi_gray.shape)


def _binarize_enhanced(enhanced: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """由 Gabor 响应图得到二值脊线（与 extract_palm_lines_binary 后半段一致）。"""
    h, w = shape[:2]
    m = min(h, w)
    block_size = max(21, int(m * 0.10) | 1)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )
    k_close = max(3, int(min(h, w) * 0.012))
    k_open = max(3, int(min(h, w) * 0.008))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(h * w * 0.004)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0
    return binary


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """骨架化：优先 ximgproc.thinning，否则形态学骨架（略粗但可用）。"""
    bin255 = ((binary > 127).astype(np.uint8) * 255)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        try:
            return cv2.ximgproc.thinning(bin255)
        except cv2.error:
            pass
    skel = np.zeros_like(bin255)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = bin255.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _mean_ridge_on_map(ridge_map: np.ndarray, cnt: np.ndarray, step: int = 4) -> float:
    """沿轮廓采样，脊线响应均值（越高越贴真实纹理）。"""
    if cnt is None or len(cnt) < 2 or ridge_map is None:
        return 0.0
    pts = cnt.reshape(-1, 2)
    h, w = ridge_map.shape[:2]
    vals = []
    for i in range(0, len(pts), step):
        x, y = int(np.clip(pts[i, 0], 0, w - 1)), int(np.clip(pts[i, 1], 0, h - 1))
        vals.append(float(ridge_map[y, x]))
    return float(np.mean(vals)) if vals else 0.0


def _refine_contour_to_ridge(ridge_map: np.ndarray, cnt: np.ndarray, win: int = 4) -> np.ndarray:
    """在脊线响应图上做小窗 argmax，把轮廓顶点吸附到局部最强纹理。"""
    if cnt is None or len(cnt) < 2 or ridge_map is None:
        return cnt
    pts = cnt.reshape(-1, 2).astype(np.float32)
    h, w = ridge_map.shape[:2]
    n = len(pts)
    stride = max(1, n // 180)
    new_pts = []
    for i in range(0, n, stride):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        x0, x1 = max(0, x - win), min(w, x + win + 1)
        y0, y1 = max(0, y - win), min(h, y + win + 1)
        patch = ridge_map[y0:y1, x0:x1]
        if patch.size == 0:
            new_pts.append([x, y])
            continue
        flat = patch.reshape(-1)
        idx = int(np.argmax(flat))
        py, px = divmod(idx, patch.shape[1])
        new_pts.append([int(x0 + px), int(y0 + py)])
    if len(new_pts) < 2:
        return cnt
    return np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)


def identify_major_lines(binary, roi_shape, ridge_map: Optional[np.ndarray] = None):
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
    skeleton = _skeletonize(binary)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return dict(life=None, head=None, heart=None)

    min_length = max(w, h) * 0.12
    valid_contours = [(cv2.arcLength(c, False), c) for c in contours
                      if cv2.arcLength(c, False) > min_length]
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    valid_contours = valid_contours[:20]

    if not valid_contours:
        return dict(life=None, head=None, heart=None)

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
        aspect = x_span / (y_span + 1e-6)
        ridge_boost = 1.0
        if ridge_map is not None:
            mr = _mean_ridge_on_map(ridge_map, cnt)
            ridge_boost = 1.0 + min(mr, 200.0) / 72.0
        candidates.append(dict(
            cnt=cnt, arc_len=arc_len,
            x_center=x_center, y_center=y_center,
            x_span=x_span, y_span=y_span,
            aspect=aspect,
            y_min=y_min, y_max=y_max,
            x_min=x_min, x_max=x_max,
            ridge_boost=ridge_boost,
        ))

    life_best = head_best = heart_best = None
    life_score = head_score = heart_score = -1.0

    for c in candidates:
        rb = c["ridge_boost"]
        # 感情线：偏上、略横向即可（放宽 aspect，更多真实弧）
        if c["y_center"] < h * 0.48 and c["aspect"] > 0.62 and c["x_span"] > w * 0.22:
            score = c["arc_len"] * (1.0 - c["y_center"] / h) * rb
            if score > heart_score:
                heart_score, heart_best = score, c["cnt"]
        # 智慧线：中部、横向为主
        if c["y_center"] > h * 0.28 and c["y_center"] < h * 0.78 and c["aspect"] > 0.52 and c["x_span"] > w * 0.22:
            score = c["arc_len"] * (1.0 - abs(c["y_center"] / h - 0.52)) * rb
            if score > head_score:
                head_score, head_best = score, c["cnt"]
        # 生命线：偏左、纵向跨度
        if c["x_center"] < w * 0.58 and c["y_span"] > h * 0.18:
            score = c["arc_len"] * (1.05 - c["x_center"] / max(w, 1)) * rb
            if score > life_score:
                life_score, life_best = score, c["cnt"]

    assigned = set()
    result = dict(life=None, head=None, heart=None)
    for name, cnt in [("heart", heart_best), ("head", head_best), ("life", life_best)]:
        if cnt is not None and id(cnt) not in assigned:
            result[name] = cnt
            assigned.add(id(cnt))

    win = int(max(3, min(h, w) * 0.022))
    win = min(win, 7)
    if ridge_map is not None:
        for key in ("heart", "head", "life"):
            if result[key] is not None:
                result[key] = _refine_contour_to_ridge(ridge_map, result[key], win=win)

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

    enhanced = enhance_palm_lines(gray)
    if enhanced is None:
        return dict(annotated=roi_bgr, binary=None, lines=None, features=None,
                    error="掌纹增强失败")
    binary = _binarize_enhanced(enhanced, gray.shape)
    lines = identify_major_lines(binary, roi_bgr.shape, ridge_map=enhanced)
    
    # 绘制线条
    annotated = draw_palm_lines_on_roi(roi_bgr, lines, thickness=2)
    
    # 提取特征
    features = extract_features_from_lines(lines, roi_bgr.shape)
    
    return dict(annotated=annotated, binary=binary, lines=lines,
                features=features, error=None)
