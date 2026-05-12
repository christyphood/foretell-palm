"""
MediaPipe-based palm line detection and drawing.

Uses MediaPipe Hand Landmarks (21 keypoints) to accurately derive and draw
the three major palm lines:
  - Life Line   (生命线): curves from between thumb/index finger down to wrist
  - Head Line   (智慧线): runs horizontally across the middle of the palm
  - Heart Line  (感情线): runs along the upper palm below the finger bases

MediaPipe 21-point landmark indices:
  0=WRIST  1=THUMB_CMC  2=THUMB_MCP  3=THUMB_IP  4=THUMB_TIP
  5=INDEX_MCP  6=INDEX_PIP  7=INDEX_DIP  8=INDEX_TIP
  9=MIDDLE_MCP 10=MIDDLE_PIP 11=MIDDLE_DIP 12=MIDDLE_TIP
  13=RING_MCP  14=RING_PIP  15=RING_DIP  16=RING_TIP
  17=PINKY_MCP 18=PINKY_PIP 19=PINKY_DIP 20=PINKY_TIP
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


# ── Landmark index constants ──────────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1
THUMB_MCP   = 2
THUMB_IP    = 3
THUMB_TIP   = 4
INDEX_MCP   = 5
INDEX_PIP   = 6
MIDDLE_MCP  = 9
MIDDLE_PIP  = 10
RING_MCP    = 13
RING_PIP    = 14
PINKY_MCP   = 17
PINKY_PIP   = 18


def _lm(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    """Convert a normalized landmark to pixel coordinates."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _cubic_bezier(p0, p1, p2, p3, n: int = 80) -> np.ndarray:
    """Sample n points on a cubic Bézier curve."""
    t = np.linspace(0.0, 1.0, n)
    pts = (
        np.outer((1 - t) ** 3, p0)
        + np.outer(3 * (1 - t) ** 2 * t, p1)
        + np.outer(3 * (1 - t) * t ** 2, p2)
        + np.outer(t ** 3, p3)
    )
    return pts.astype(np.int32)


# ── Palm line derivation ──────────────────────────────────────────────────────

def _life_line(lm, w, h) -> np.ndarray:
    """
    生命线: starts at the web between thumb and index finger,
    curves around the thenar eminence down to the wrist.
    """
    thumb_mcp = _lm(lm, THUMB_MCP, w, h)
    thumb_cmc = _lm(lm, THUMB_CMC, w, h)
    index_mcp = _lm(lm, INDEX_MCP, w, h)
    wrist     = _lm(lm, WRIST,     w, h)

    p0 = _lerp(thumb_mcp, index_mcp, 0.5)   # start: web of thumb/index
    p1 = _lerp(thumb_cmc, index_mcp, 0.4)   # upper control
    p2 = _lerp(thumb_cmc, wrist,     0.5)   # lower control
    p3 = wrist                               # end: wrist centre

    return _cubic_bezier(p0, p1, p2, p3)


def _head_line(lm, w, h) -> np.ndarray:
    """
    智慧线: starts near the life-line origin, runs horizontally
    across the middle of the palm toward the pinky side.
    """
    thumb_mcp  = _lm(lm, THUMB_MCP,  w, h)
    index_mcp  = _lm(lm, INDEX_MCP,  w, h)
    middle_mcp = _lm(lm, MIDDLE_MCP, w, h)
    ring_mcp   = _lm(lm, RING_MCP,   w, h)
    pinky_mcp  = _lm(lm, PINKY_MCP,  w, h)
    wrist      = _lm(lm, WRIST,      w, h)

    palm_h = float(np.linalg.norm(middle_mcp - wrist))
    down   = np.array([0.0, palm_h * 0.28])

    p0 = _lerp(thumb_mcp, index_mcp, 0.5)
    p1 = index_mcp  + down * 0.55
    p2 = ring_mcp   + down * 0.90
    p3 = pinky_mcp  + down * 0.70

    return _cubic_bezier(p0, p1, p2, p3)


def _heart_line(lm, w, h) -> np.ndarray:
    """
    感情线: runs just below the finger bases from the index/middle
    side across to the pinky.
    """
    index_mcp  = _lm(lm, INDEX_MCP,  w, h)
    middle_mcp = _lm(lm, MIDDLE_MCP, w, h)
    ring_mcp   = _lm(lm, RING_MCP,   w, h)
    pinky_mcp  = _lm(lm, PINKY_MCP,  w, h)
    wrist      = _lm(lm, WRIST,      w, h)

    palm_h = float(np.linalg.norm(middle_mcp - wrist))
    down   = np.array([0.0, palm_h * 0.11])

    p0 = index_mcp  + down
    p1 = middle_mcp + down * 0.85
    p2 = ring_mcp   + down * 1.05
    p3 = pinky_mcp  + down * 0.80

    return _cubic_bezier(p0, p1, p2, p3)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_line(canvas: np.ndarray, pts: np.ndarray,
               color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """Draw an anti-aliased polyline with a subtle glow effect."""
    # Glow pass (wider, semi-transparent)
    glow = canvas.copy()
    cv2.polylines(glow, [pts.reshape(-1, 1, 2)], False,
                  color, thickness + 4, cv2.LINE_AA)
    canvas = cv2.addWeighted(glow, 0.30, canvas, 0.70, 0)
    # Main line
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False,
                  color, thickness, cv2.LINE_AA)
    return canvas


# ── Feature extraction ────────────────────────────────────────────────────────

def _arc_length(pts: np.ndarray) -> float:
    diffs = np.diff(pts.astype(np.float64), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _curvature(pts: np.ndarray) -> float:
    arc = _arc_length(pts)
    straight = float(np.linalg.norm(pts[-1].astype(float) - pts[0].astype(float)))
    return arc / straight if straight > 1 else 1.0


def _angle(pts: np.ndarray) -> float:
    d = pts[-1].astype(float) - pts[0].astype(float)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _approx_intersections(a: np.ndarray, b: np.ndarray, tol: int = 10) -> int:
    count = 0
    for pa in a[::5]:
        if np.linalg.norm(b.astype(float) - pa.astype(float), axis=1).min() < tol:
            count += 1
    return min(count, 5)


def _extract_features(life: np.ndarray, head: np.ndarray,
                      heart: np.ndarray) -> Dict:
    f = {}
    for name, pts in [("life", life), ("head", head), ("heart", heart)]:
        f[f"{name}_length"]    = _arc_length(pts)
        f[f"{name}_curvature"] = _curvature(pts)
        f[f"{name}_angle"]     = _angle(pts)
    f["life_head_intersection"]  = _approx_intersections(life, head)
    f["life_heart_intersection"] = _approx_intersections(life, heart)
    f["head_heart_intersection"] = _approx_intersections(head, heart)
    f["line_coverage"] = 0.0
    return f


# ── Public API ────────────────────────────────────────────────────────────────

def detect_and_draw_palm_lines(
    image_rgb: np.ndarray,
    draw_skeleton: bool = False,
    line_thickness: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[str]]:
    """
    Detect hand landmarks with MediaPipe and draw the three major palm lines.

    Args:
        image_rgb:      H×W×3 RGB uint8 numpy array.
        draw_skeleton:  Also draw the 21 MediaPipe keypoints and connections.
        line_thickness: Pixel thickness of the drawn palm lines.

    Returns:
        (overlay, features, error)
        overlay  — annotated RGB image, or None on failure.
        features — dict of geometric features, or None on failure.
        error    — human-readable error string, or None on success.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None, None, "MediaPipe 未安装，请运行: pip install mediapipe"

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    h, w = image_rgb.shape[:2]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    ) as hands:
        result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None, None, (
            "未检测到手掌。请确保手掌正面朝向镜头、光线充足、手指展开。"
        )

    lm     = result.multi_hand_landmarks[0].landmark
    canvas = image_rgb.copy()

    # Optionally draw skeleton
    if draw_skeleton:
        mp_drawing.draw_landmarks(
            canvas,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1),
        )

    # Derive palm line polylines
    life_pts  = _life_line(lm,  w, h)
    head_pts  = _head_line(lm,  w, h)
    heart_pts = _heart_line(lm, w, h)

    # Draw lines  (Life=Red, Head=Green, Heart=Blue)
    canvas = _draw_line(canvas, life_pts,  (220,  55,  55), line_thickness)
    canvas = _draw_line(canvas, head_pts,  ( 55, 200,  75), line_thickness)
    canvas = _draw_line(canvas, heart_pts, ( 60, 120, 220), line_thickness)

    # Endpoint dots
    for pts, col in [(life_pts, (220, 55, 55)),
                     (head_pts, (55, 200, 75)),
                     (heart_pts, (60, 120, 220))]:
        cv2.circle(canvas, tuple(pts[0]),  6, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), 6, col, -1, cv2.LINE_AA)

    features = _extract_features(life_pts, head_pts, heart_pts)
    return canvas, features, None
