# -*- coding: utf-8 -*-
"""
Palm ROI Extractor — 移植自 Huterox/palm_recongnition
https://github.com/Huterox/palm_recongnition

核心流程（与原项目一致）：
  1. 手部关键点检测（OpenCV DNN + MediaPipe hand_landmark.tflite）
  2. 图像旋转矫正（腕部→中指MCP向量对齐Y轴）
  3. 手部分割（GrabCut，替代原项目的 ModelScope ResNet50）
  4. PSO 算法精确定位掌心圆心和最大内接圆半径
  5. 提取 ROI（正方形 + 圆形）
"""

import cv2
import numpy as np
import os
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# 1. PSO 算法（直接移植自 palm_roi_ext/palm_core/position_fitness/pso.py）
# ─────────────────────────────────────────────────────────────────────────────

class PSOInstance:
    def __init__(self, num_particles, center, radius, max_iterations,
                 bounds, fitness_function,
                 w_start=0.9, w_end=0.4, c1=1.496, c2=1.496):
        self.num_particles   = num_particles
        self.center          = np.array(center, dtype=float)
        self.radius          = radius
        self.max_iterations  = max_iterations
        self.bounds          = np.array(bounds, dtype=float)
        self.fitness_function = fitness_function
        self.w_start = w_start
        self.w_end   = w_end
        self.c1 = c1
        self.c2 = c2
        self.population = self._init_population()

    def _init_population(self):
        dim = len(self.bounds)
        population = []
        self.global_best_fitness  = float('inf')
        self.global_best_position = None
        for _ in range(self.num_particles):
            pos = self.center + (np.random.rand(dim) * 2 - 1) * self.radius
            particle = dict(position=pos, velocity=np.zeros(dim),
                            best_position=pos.copy(), best_fitness=None)
            fitness = self.fitness_function(pos)
            particle['best_fitness'] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness  = fitness
                self.global_best_position = pos.copy()
            population.append(particle)
        return population

    def optimize(self):
        for it in range(self.max_iterations):
            w = self.w_start - (self.w_start - self.w_end) * (it / self.max_iterations)
            for p in self.population:
                cog = self.c1 * np.random.rand() * (p['best_position'] - p['position'])
                soc = self.c2 * np.random.rand() * (self.global_best_position - p['position'])
                p['velocity']  = w * p['velocity'] + cog + soc
                p['position'] += p['velocity']
                p['position']  = np.clip(p['position'], self.bounds[:, 0], self.bounds[:, 1])
                fitness = self.fitness_function(p['position'])
                if p['best_fitness'] is None or fitness < p['best_fitness']:
                    p['best_fitness']  = fitness
                    p['best_position'] = p['position'].copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness  = fitness
                    self.global_best_position = p['position'].copy()

    def get_best_solution(self):
        return self.global_best_position, self.global_best_fitness


# ─────────────────────────────────────────────────────────────────────────────
# 2. PSO 掌心定位（移植自 palm_pso.py）
# ─────────────────────────────────────────────────────────────────────────────

class PsoPositionPalm:
    """用 PSO 在二值手掌图像中寻找最大内接圆（掌心 ROI）。"""

    def __init__(self):
        self.random_radius    = 5
        self.population_number = 5
        self.iter_number      = 30
        self.padding_step     = 2

    def fitness(self, center):
        """返回负的最大内接圆半径（PSO 最小化目标）。"""
        h, w = self.binary_image.shape
        cx, cy = int(center[0]), int(center[1])
        max_r = self.base_radius
        for r in range(self.base_radius, max(h, w), self.padding_step):
            if cx - r < 0 or cx + r >= w or cy - r < 0 or cy + r >= h:
                break
            hit_bg = False
            for angle in range(0, 360, 10):
                x = int(cx + r * np.cos(np.deg2rad(angle)))
                y = int(cy + r * np.sin(np.deg2rad(angle)))
                if not (0 <= x < w and 0 <= y < h):
                    hit_bg = True
                    break
                if self.binary_image[y, x] == 0:
                    hit_bg = True
                    break
            if hit_bg:
                break
            max_r = r
        return -max_r

    def optimize(self, center, binary_image, x_up, y_up, base_radius):
        self.binary_image = binary_image
        self.base_radius  = base_radius
        pso = PSOInstance(
            num_particles    = self.population_number,
            center           = center,
            radius           = self.random_radius,
            max_iterations   = self.iter_number,
            bounds           = [[0, x_up], [0, y_up]],
            fitness_function = self.fitness,
        )
        pso.optimize()
        best_pos, best_fit = pso.get_best_solution()
        return [int(best_pos[0]), int(best_pos[1])], abs(int(best_fit))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 图像旋转矫正（移植自 rotate.py）
# ─────────────────────────────────────────────────────────────────────────────

def rotate_hand_image(key_points, image):
    """
    根据腕部(0)→中指MCP(9)向量将手掌旋转至竖直方向。
    返回 (rotated_image, angle_deg, rotated_key_points)
    """
    wrist  = key_points[0]
    mid9   = key_points[9]
    h, w   = image.shape[:2]

    vx = mid9[0] - wrist[0]
    vy = -(mid9[1] - wrist[1])          # 翻转 y 轴

    vec_xoy = np.array([w, 0], dtype=float)
    vec     = np.array([vx, vy], dtype=float)
    cos_a   = np.dot(vec, vec_xoy) / (np.linalg.norm(vec) * np.linalg.norm(vec_xoy) + 1e-8)
    angle   = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    y9, y0 = key_points[9][1], key_points[0][1]
    if y0 > y9:
        angle = 90 - angle if angle <= 90 else -(angle - 90)
    else:
        angle = angle + 90 if angle <= 90 else -(180 - (angle - 90))

    center = (w / 2, h / 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    rot_pts = []
    for pt in key_points:
        hp = np.array([pt[0], pt[1], 1.0])
        tp = M @ hp
        rot_pts.append(tp[:2])
    return rotated, angle, np.array(rot_pts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 手部分割（GrabCut，替代原项目 ModelScope ResNet50）
# ─────────────────────────────────────────────────────────────────────────────

def segment_hand_grabcut(image, key_points=None):
    """
    用 GrabCut 分割手部区域，返回 (hand_only_image, binary_mask)。
    如果提供了关键点，用关键点的包围盒初始化 GrabCut 矩形。
    """
    h, w = image.shape[:2]

    if key_points is not None and len(key_points) > 0:
        xs = np.clip([int(p[0]) for p in key_points], 0, w - 1)
        ys = np.clip([int(p[1]) for p in key_points], 0, h - 1)
        pad = int(min(w, h) * 0.05)
        x1 = max(0, xs.min() - pad)
        y1 = max(0, ys.min() - pad)
        x2 = min(w - 1, xs.max() + pad)
        y2 = min(h - 1, ys.max() + pad)
        rect = (x1, y1, x2 - x1, y2 - y1)
    else:
        margin = int(min(w, h) * 0.05)
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    mask_gc = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # GrabCut 失败时退化为简单阈值分割
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hand_only = cv2.bitwise_and(image, image,
                                    mask=(binary > 0).astype(np.uint8) * 255)
        return hand_only, binary

    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    hand_only = cv2.bitwise_and(image, image, mask=fg_mask)
    return hand_only, fg_mask


# ─────────────────────────────────────────────────────────────────────────────
# 5. 初始化圆心（移植自 positions.py TriangleTransform）
# ─────────────────────────────────────────────────────────────────────────────

def get_init_center_from_keypoints(key_points):
    """
    用关键点 0（腕）、5（食指MCP）、13（无名指MCP）的三角形重心初始化圆心。
    返回 (center, base_radius)
    """
    pts = [key_points[0], key_points[5], key_points[13]]
    cx  = sum(p[0] for p in pts) / 3
    cy  = sum(p[1] for p in pts) / 3
    # 基础半径：小指MCP 到圆心的水平距离的一半
    base_r = abs(int((key_points[17][0] - cx) / 2))
    base_r = max(base_r, 5)
    return (cx, cy), base_r


def get_init_center_from_binary(binary_image):
    """
    当关键点不可用时，用距离变换找掌心（移植自 DistTransform）。
    """
    dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc, 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. 手部关键点检测（OpenCV DNN + MediaPipe hand_landmark ONNX）
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
_ONNX_URL   = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx"
_ONNX_PATH  = os.path.join(_MODEL_DIR, "hand_landmark.onnx")

_ort_session = None   # 懒加载


def _ensure_model():
    """下载并缓存 MediaPipe hand landmark ONNX 模型，用 onnxruntime 加载。"""
    global _ort_session
    if _ort_session is not None:
        return True
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[palm_roi] onnxruntime 未安装，请运行: pip install onnxruntime")
        return False
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_ONNX_PATH):
        print(f"[palm_roi] 正在下载手部关键点模型...")
        try:
            urllib.request.urlretrieve(_ONNX_URL, _ONNX_PATH)
            print(f"[palm_roi] 模型下载完成: {_ONNX_PATH}")
        except Exception as e:
            print(f"[palm_roi] 模型下载失败: {e}")
            return False
    try:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(_ONNX_PATH)
        return True
    except Exception as e:
        print(f"[palm_roi] 模型加载失败: {e}")
        return False


def detect_hand_keypoints_dnn(image_bgr):
    """
    用 onnxruntime 运行 MediaPipe hand landmark ONNX 模型。
    模型输入: [1, 224, 224, 3] float32 (NHWC, 0~1)
    模型输出: Identity [1, 63] — 21 个关键点 (x, y, z) 归一化坐标
    返回 shape (21, 2) 的像素坐标数组，失败返回 None。
    """
    if not _ensure_model():
        return None
    h, w = image_bgr.shape[:2]

    # 预处理：resize → RGB → [0,1] → NHWC batch
    img_resized = cv2.resize(image_bgr, (224, 224))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob        = img_rgb[np.newaxis, ...]   # shape: (1, 224, 224, 3)

    try:
        outputs = _ort_session.run(None, {"input_1": blob})
    except Exception as e:
        print(f"[palm_roi] 推理失败: {e}")
        return None

    # outputs[0] = Identity: [1, 63] — 21 landmarks (x, y, z)
    out = outputs[0].flatten()
    if len(out) < 63:
        return None
    pts = out[:63].reshape(21, 3)
    # 坐标在 224×224 空间中，映射到原图尺寸
    pts[:, 0] = pts[:, 0] / 224.0 * w
    pts[:, 1] = pts[:, 1] / 224.0 * h
    return pts[:, :2]   # (21, 2) 像素坐标


# ─────────────────────────────────────────────────────────────────────────────
# 7. 主入口：ROI 提取（对应原项目 AutoRotateRoIExtract.roi_extract）
# ─────────────────────────────────────────────────────────────────────────────

def extract_palm_roi(image_bgr):
    """
    完整 ROI 提取流程（移植自 Huterox/palm_recongnition AutoRotateRoIExtract）。

    Args:
        image_bgr: BGR numpy 图像

    Returns:
        dict with keys:
          'draw_img'    — 标注了 ROI 圆和矩形的图像
          'roi_square'  — 裁剪的正方形 ROI
          'roi_circle'  — 裁剪的圆形 ROI（黑色背景）
          'center'      — (cx, cy) 掌心像素坐标
          'radius'      — 最大内接圆半径
          'rotated_img' — 旋转矫正后的图像
          'key_points'  — (21,2) 关键点数组，或 None
          'error'       — 错误信息字符串，或 None
    """
    # 缩放大图以加速处理（保持宽高比，最大边 800px）
    h_orig, w_orig = image_bgr.shape[:2]
    max_dim = 800
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = image_bgr.shape[:2]
    x_up, y_up = w, h

    # ── Step 1: 关键点检测 ────────────────────────────────────────────────────
    key_points = detect_hand_keypoints_dnn(image_bgr)
    has_kp = key_points is not None

    # ── Step 2: 旋转矫正 ──────────────────────────────────────────────────────
    if has_kp:
        rotated, angle, key_points = rotate_hand_image(key_points, image_bgr)
    else:
        rotated = image_bgr.copy()

    # ── Step 3: 手部分割 ──────────────────────────────────────────────────────
    hand_only, binary = segment_hand_grabcut(rotated, key_points if has_kp else None)

    if binary is None or binary.max() == 0:
        return dict(draw_img=image_bgr, roi_square=None, roi_circle=None,
                    center=None, radius=None, rotated_img=rotated,
                    key_points=key_points,
                    error="手部分割失败，请确保手掌清晰可见。")

    # ── Step 4: 初始化圆心 ────────────────────────────────────────────────────
    if has_kp:
        center, base_radius = get_init_center_from_keypoints(key_points)
    else:
        center, base_radius = get_init_center_from_binary(binary)

    # ── Step 5: PSO 优化圆心和半径 ────────────────────────────────────────────
    pso = PsoPositionPalm()
    center, radius = pso.optimize(center, binary, x_up, y_up, base_radius)

    if radius < 10:
        return dict(draw_img=hand_only, roi_square=None, roi_circle=None,
                    center=tuple(center), radius=radius, rotated_img=rotated,
                    key_points=key_points,
                    error="PSO 未能找到有效的掌心区域，请尝试更清晰的图片。")

    # ── Step 6: 提取 ROI ──────────────────────────────────────────────────────
    draw_img, roi_square, roi_circle = _extract_roi(hand_only, center, radius)

    return dict(draw_img=draw_img, roi_square=roi_square, roi_circle=roi_circle,
                center=tuple(center), radius=radius, rotated_img=rotated,
                key_points=key_points, error=None)


def _extract_roi(image_rgb, center, max_radius):
    """
    在图像上绘制最大内接圆和内切正方形，并裁剪 ROI。
    移植自 ROIExtract.extract_roi。
    """
    cx, cy = int(center[0]), int(center[1])
    r      = int(max_radius)

    # 绘制圆和正方形
    vis = image_rgb.copy()
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
    side = int(r * np.sqrt(2))
    tl   = (cx - side // 2, cy - side // 2)
    br   = (cx + side // 2, cy + side // 2)
    cv2.rectangle(vis, tl, br, (0, 0, 255), 2)
    cv2.circle(vis, (cx, cy), 5, (255, 50, 60), -1)

    # 裁剪正方形
    x1 = max(0, tl[0]);  y1 = max(0, tl[1])
    x2 = min(image_rgb.shape[1], br[0])
    y2 = min(image_rgb.shape[0], br[1])
    roi_square = image_rgb[y1:y2, x1:x2].copy()

    # 裁剪圆形（黑色背景）
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    circle_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    nzy, nzx   = np.nonzero(mask)
    if len(nzy) > 0:
        roi_circle = circle_img[nzy.min():nzy.max(), nzx.min():nzx.max()].copy()
    else:
        roi_circle = roi_square.copy()

    return vis, roi_square, roi_circle
