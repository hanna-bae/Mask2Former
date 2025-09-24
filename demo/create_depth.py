#!/usr/bin/env python3
"""
DepthPro Inference & Visualization
참고: Apple DepthPro 샘플 스크립트
기능:
- 입력 이미지(단일 or 폴더)에 대해 DepthPro 추론
- raw depth 저장 (.npz)
- colormap depth 시각화 저장 (.jpg)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from ml_depth_pro.src.depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Select available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def run_inference(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load DepthPro
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    # Collect images
    if args.image_path.is_dir():
        image_paths = list(args.image_path.glob("**/*"))
        relative_path = args.image_path
    else:
        image_paths = [args.image_path]
        relative_path = args.image_path.parent

    for image_path in tqdm(image_paths):
        try:
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            LOGGER.warning(f"Skipping {image_path}: {e}")
            continue

        # 결과 파일 경로 (jpg만)
        output_file = (
            args.output_path
            / image_path.relative_to(relative_path).parent
            / image_path.stem
        )
        jpg_out = output_file.with_suffix(".jpg")

        # jpg가 이미 있으면 inference 스킵
        if jpg_out.exists():
            LOGGER.info(f"Skipping inference for {image_path}, {jpg_out} already exists.")
            continue

        # ----- Inference -----
        prediction = model.infer(transform(image), f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()

        # Inverse depth normalization for visualization
        inv_depth = 1.0 / np.clip(depth, 1e-3, 1e3)
        max_inv = min(inv_depth.max(), 1 / 0.1)   # clip 0.1m
        min_inv = max(inv_depth.min(), 1 / 250)   # clip 250m
        inv_norm = (inv_depth - min_inv) / (max_inv - min_inv + 1e-6)

        # 저장 경로 생성
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save visualization (jpg만)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inv_norm)[..., :3] * 255).astype(np.uint8)
        PIL.Image.fromarray(color_depth).save(jpg_out, format="JPEG", quality=95)

        LOGGER.info(f"Saved: {jpg_out}")



def main():
    parser = argparse.ArgumentParser(description="DepthPro visualization script")
    parser.add_argument(
        "-i", "--image-path",
        type=Path,
        required=True,
        help="Path to input image or folder"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=Path,
        required=True,
        help="Path to save depth results"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()