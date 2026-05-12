#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试掌纹分析系统的核心功能
"""

import sys
import cv2
import numpy as np
from utils.palm_roi_extractor import extract_palm_roi
from utils.palm_line_drawer   import process_palm_lines


def test_with_sample_image():
    """用项目中的样本图片测试完整流程"""
    
    # 尝试加载样本图片
    sample_paths = [
        "data/images/20251128_085131.jpg",
        "data/images/20251128_085137.jpg",
        "data/images/IMG-20251128-WA0004.jpg",
    ]
    
    img = None
    for path in sample_paths:
        try:
            img = cv2.imread(path)
            if img is not None:
                print(f"✅ 加载图片: {path}")
                break
        except Exception as e:
            continue
    
    if img is None:
        print("❌ 未找到样本图片，请确保 data/images/ 目录下有图片")
        return False
    
    print(f"   图片尺寸: {img.shape}")
    
    # ── Step 1: ROI 提取 ──────────────────────────────────────────────────────
    print("\n[1/2] 正在提取掌心 ROI...")
    roi_result = extract_palm_roi(img)
    
    if roi_result['error'] is not None:
        print(f"❌ ROI 提取失败: {roi_result['error']}")
        return False
    
    print(f"✅ ROI 提取成功")
    print(f"   掌心坐标: {roi_result['center']}")
    print(f"   ROI 半径: {roi_result['radius']} 像素")
    
    # 保存 ROI 标注图
    cv2.imwrite("/tmp/palm_roi_annotated.jpg", roi_result['draw_img'])
    print(f"   已保存: /tmp/palm_roi_annotated.jpg")
    
    if roi_result['roi_square'] is not None:
        cv2.imwrite("/tmp/palm_roi_square.jpg", roi_result['roi_square'])
        print(f"   已保存: /tmp/palm_roi_square.jpg")
    
    # ── Step 2: 掌纹线条提取 ──────────────────────────────────────────────────
    print("\n[2/2] 正在提取掌纹线条...")
    line_result = process_palm_lines(roi_result['roi_square'])
    
    if line_result['error'] is not None:
        print(f"❌ 掌纹提取失败: {line_result['error']}")
        return False
    
    print(f"✅ 掌纹提取成功")
    
    # 显示特征
    features = line_result['features']
    print(f"\n   特征数据:")
    print(f"   - 生命线长度: {features['life_length']:.1f} px")
    print(f"   - 智慧线长度: {features['head_length']:.1f} px")
    print(f"   - 感情线长度: {features['heart_length']:.1f} px")
    print(f"   - 生命线曲率: {features['life_curvature']:.3f}")
    print(f"   - 智慧线曲率: {features['head_curvature']:.3f}")
    print(f"   - 感情线曲率: {features['heart_curvature']:.3f}")
    
    # 保存结果
    if line_result['annotated'] is not None:
        cv2.imwrite("/tmp/palm_lines_annotated.jpg", line_result['annotated'])
        print(f"\n   已保存: /tmp/palm_lines_annotated.jpg")
    
    if line_result['binary'] is not None:
        cv2.imwrite("/tmp/palm_lines_binary.jpg", line_result['binary'])
        print(f"   已保存: /tmp/palm_lines_binary.jpg")
    
    print("\n✅ 测试完成！")
    return True


if __name__ == "__main__":
    success = test_with_sample_image()
    sys.exit(0 if success else 1)
