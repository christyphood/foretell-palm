# -*- coding: utf-8 -*-
"""
可选集成 EDCC（Enhanced and Discriminative Competitive Code）。

EDCC 来自: https://github.com/leosocy/EDCC-Palmprint-Recognition
用途是「掌纹编码 + 两图相似度」，用于身份比对类场景，并不从单张图输出三条感情/智慧/生命线。
需在系统内先按上游文档编译安装 libedcc，再安装 pypackage 后 `import edcc` 才会生效。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import cv2
import numpy as np


def collect_edcc_optional(
    roi_main_bgr: np.ndarray,
    ref_roi_bgr: Optional[np.ndarray] = None,
    ref_decode_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对 ROI 掌纹区域做 EDCC 编码；若提供第二张 ROI 则计算 competitive code 相似度。

    roi_* 为 BGR uint8。
    """
    doc = "https://github.com/leosocy/EDCC-Palmprint-Recognition"
    out: Dict[str, Any] = {
        "library": "leosocy/EDCC-Palmprint-Recognition",
        "doc": doc,
        "available": False,
        "role": "encoding_and_pairwise_similarity",
        "note_zh": "EDCC 用于掌纹编码与两图比对，不替代 Gabor/骨架化主线提取。",
    }
    if ref_decode_error:
        out["reference_error"] = ref_decode_error
    try:
        import edcc  # type: ignore
    except Exception as e:  # noqa: BLE001
        out["error"] = f"edcc 未安装或 native 库不可用: {e}"
        out["install_hint"] = "clone 上游仓库后 cmake 安装 libedcc，再 cd pypackage && pip install ."
        return out

    config = edcc.EncoderConfig(29, 5, 5, 10)
    encoder = edcc.create_encoder(config)

    def _encode(roi: np.ndarray):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ok, buf = cv2.imencode(".bmp", gray)
        if not ok:
            raise RuntimeError("ROI 无法编码为 BMP")
        return encoder.encode_using_bytes(buf.tobytes())

    try:
        code_main = _encode(roi_main_bgr)
        raw = getattr(code_main, "_code", None)
        if raw is not None:
            out["code_digest_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
            out["code_length_bytes"] = len(raw)
        out["available"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out

    if ref_roi_bgr is not None and ref_roi_bgr.size > 0:
        try:
            code_ref = _encode(ref_roi_bgr)
            out["similarity_to_reference"] = float(code_main.compare_to(code_ref))
        except Exception as e:  # noqa: BLE001
            out["compare_error"] = str(e)
    return out
