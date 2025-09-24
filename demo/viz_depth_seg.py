# -*- coding: utf-8 -*-
"""
입력 폴더의 JPG 이미지를 순회하며
1) Mask2Former 세그멘테이션 시각화 결과(.png)
2) Zero-shot DepthPro 딥스맵(.npy + 컬러맵 .png)
를 저장합니다.

Python 3.8+ 호환 (typing은 Optional/Union 사용)
"""

import os
import argparse
import glob
import tempfile
from typing import Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# repo 루트 기준에서 실행한다고 가정 (mask2former & predictor 모듈 사용)
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

# DepthPro
from ml_depth_pro.src.depth_pro import create_model_and_transforms, load_rgb


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def inv_depth_turbo(depth_m,
                    near=0.1,
                    far=250.0,
                    gamma=0.85,
                    clip_percentile=None):
    """
    depth_m: (H,W) float32, meters
    near/far: 시각화용 클리핑 범위
    gamma: 0.7~1.0 권장(0.85 기본)  -> 노랑/청록 대비 보정
    clip_percentile: (lo, hi) 퍼센타일로 동적 클리핑. 예: (1, 99). None이면 사용 안 함.

    반환: BGR uint8 (OpenCV 저장용)
    """
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = np.inf

    if clip_percentile is not None:
        lo, hi = clip_percentile
        # 유효치만 퍼센타일 계산
        valid = np.isfinite(d) & (d < np.inf)
        if valid.any():
            lo_v = np.percentile(d[valid], lo)
            hi_v = np.percentile(d[valid], hi)
            near, far = max(near, lo_v), min(far, hi_v)

    # 역깊이 정규화 [1/far, 1/near] -> [0,1]
    d = np.clip(d, near, far)
    inv = 1.0 / d
    inv_min = 1.0 / float(far)
    inv_max = 1.0 / float(near)
    denom = inv_max - inv_min if (inv_max - inv_min) > 1e-6 else 1e-6
    inv_norm = (inv - inv_min) / denom
    inv_norm = np.clip(inv_norm, 0.0, 1.0)

    # 감마 보정(밝은 영역 확장: 노랑/청록 더 또렷)
    if gamma is not None and gamma > 0:
        inv_norm = np.power(inv_norm, gamma)

    # 8-bit로 변환 후 Turbo 컬러맵
    inv_u8 = (inv_norm * 255.0).astype(np.uint8)
    if hasattr(cv2, "COLORMAP_TURBO"):
        color_bgr = cv2.applyColorMap(inv_u8, cv2.COLORMAP_TURBO)
    else:
        # OpenCV에 Turbo 없으면 Jet(가까운 대체) 사용
        color_bgr = cv2.applyColorMap(inv_u8, cv2.COLORMAP_JET)

    return color_bgr


_DEPTH_CACHE = {"model": None, "transform": None, "device": None}


def get_depth_model(device, precision):
    if _DEPTH_CACHE["model"] is None:
        model, transform = create_model_and_transforms(device=device, precision=precision)
        model.eval()
        _DEPTH_CACHE.update({"model": model, "transform": transform, "device": device})
    return _DEPTH_CACHE["model"], _DEPTH_CACHE["transform"], _DEPTH_CACHE["device"]


def run_depth(image_path, device, precision ):
    """
    DepthPro 추론. 반환: depth (H,W) float32, meters
    """
    img, _, f_px = load_rgb(image_path)  # img: PIL Image (RGB), f_px: focal length px
    model, transform, _ = get_depth_model(device=device, precision=precision)
    with torch.no_grad():
        pred = model.infer(transform(img), f_px=f_px)
    depth = pred["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
    return depth


def save_image_bgr(path, rgb_img):
    """
    detectron2 vis 결과는 RGB 이므로 OpenCV 저장 위해 BGR로 변환
    """
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    parser = argparse.ArgumentParser(description="Export segmentation (Mask2Former) and depth (DepthPro) from JPG folder.")
    # I/O
    parser.add_argument("--input_dir", type=str, required=True, help="입력 JPG 폴더")
    parser.add_argument("--output_dir", type=str, required=True, help="결과 저장 폴더")
    parser.add_argument("--ext", type=str, default="jpg", help="이미지 확장자 (기본: jpg)")

    # Detectron2 / Mask2Former
    parser.add_argument("--config-file", type=str,
                        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
                        help="Mask2Former config 파일")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="시각화 표시 임계값")
    parser.add_argument("--opts", default=["MODEL.WEIGHTS", "./ckpt/model_final_dfa862.pkl"],
                        nargs=argparse.REMAINDER, help="cfg override 'KEY VALUE' pairs")

    # Depth
    parser.add_argument("--depth-device", type=str, default="cuda:0", help="DepthPro 디바이스 (예: cuda:0 / cpu)")
    parser.add_argument("--depth-precision", type=str, default="fp16", choices=["fp16", "fp32"], help="DepthPro 추론 정밀도")
    parser.add_argument("--depth-near", type=float, default=0.1, help="inverse-depth near clip (m)")
    parser.add_argument("--depth-far", type=float, default=250.0, help="inverse-depth far clip (m)")
    parser.add_argument("--depth-gamma", type=float, default=0.85, help="inverse-depth gamma")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seg_dir = os.path.join(args.output_dir, "seg_vis")
    dep_dir = os.path.join(args.output_dir, "depth")
    dep_png_dir = os.path.join(args.output_dir, "depth_colormap")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)
    os.makedirs(dep_png_dir, exist_ok=True)

    # Logger
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Starting...")

    # Setup Detectron2 predictor (Mask2Former)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, instance_mode=None, parallel=False)
    demo.predictor.confidence_threshold = args.confidence_threshold

    # Depth precision
    prec = torch.half if args.depth_precision == "fp16" else torch.float32

    # Gather images
    pat = os.path.join(args.input_dir, f"**/*.{args.ext}")
    img_paths = sorted(glob.glob(pat, recursive=True))
    if len(img_paths) == 0:
        logger.error(f"No '*.{args.ext}' found under {args.input_dir}")
        return

    for img_path in tqdm(img_paths, desc="Processing"):
        name = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- Segmentation (Mask2Former) ----------
        # read_image -> RGB ndarray
        image = read_image(img_path, format="RGB")
        predictions, visualized_output = demo.run_on_image(image)
        seg_rgb = visualized_output.get_image()  # RGB ndarray
        save_image_bgr(os.path.join(seg_dir, f"{name}_seg.png"), seg_rgb)

        # ---------- Depth (DepthPro) ----------
        depth_m = run_depth(img_path, device=args.depth_device, precision=prec)

        # raw depth 저장 (.npy)
        np.save(os.path.join(dep_dir, f"{name}_depth.npy"), depth_m)

        # 컬러맵 depth 저장 (.png)
        depth_color = inv_depth_turbo(
    depth_m,
    near=args.depth_near if hasattr(args, "depth_near") else 0.1,
    far=args.depth_far if hasattr(args, "depth_far") else 250.0,
    gamma=0.85,
    clip_percentile=None  # 예: (1, 99)로 주면 씬마다 자동 대비
)
        cv2.imwrite(os.path.join(dep_png_dir, f"{name}_depth.png"), depth_color)

    logger.info(f"Done. seg_vis={seg_dir}, depth_raw={dep_dir}, depth_colormap={dep_png_dir}")


if __name__ == "__main__":
    # 멀티프로세싱 이슈 회피용 (detectron2/torch)
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
