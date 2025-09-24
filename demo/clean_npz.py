#!/usr/bin/env python3
import os
from pathlib import Path

# 정리할 최상위 경로 (output-path 기준)
TARGET_DIR = Path("./data/vod/view_of_delft_PUBLIC/radar_5frames/training/depth_viz/")  # <-- 본인 output 경로로 수정

count = 0
for npz_file in TARGET_DIR.rglob("*.npz"):
    try:
        npz_file.unlink()
        count += 1
        print(f"Deleted: {npz_file}")
    except Exception as e:
        print(f"Failed to delete {npz_file}: {e}")

print(f"총 {count} 개의 .npz 파일을 삭제했습니다.")
