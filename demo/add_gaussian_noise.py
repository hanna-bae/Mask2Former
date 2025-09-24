from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm 

def add_gaussian_noise(img, rho, sigma, seed):
    """
    img: np.ndarray, shape (H,W) or (H,W,C), dtype uint8/float32 ...
    rho: noise level (하이퍼파라미터)
    sigma: base std in normalized [0,1] space. 보통 1.0로 고정하고 rho만 조절.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # to float [0,1]
    orig_dtype = img.dtype
    if np.issubdtype(orig_dtype, np.integer):
        img_f = img.astype(np.float32) / 255.0
        scale_back = 255.0
        out_dtype = np.uint8
    else:
        # assume already [0,1] float
        img_f = img.astype(np.float32)
        scale_back = 1.0
        out_dtype = img.dtype

    noise = rng.normal(loc=0.0, scale=sigma, size=img_f.shape).astype(np.float32)
    noisy = img_f + rho * noise
    noisy = np.clip(noisy, 0.0, 1.0)

    # back to original range/dtype
    noisy = (noisy * scale_back).round() if out_dtype == np.uint8 else noisy
    return noisy.astype(out_dtype)

def load_image(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return arr

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def main():
    parser = argparse.ArgumentParser(description="Vision Failure Test: add Gaussian noise to images")
    parser.add_argument("--input", type=str, required=True, help="image file or directory")
    parser.add_argument("--output", type=str, required=True, help="output file or directory")
    parser.add_argument("--rho", type=float, default=None,
                        help="noise level ρ. 예: 0.05, 0.1, 0.2 ...")
    parser.add_argument("--rhos", type=float, nargs="*", default=None,
                        help="여러 강도를 한 번에. 예: --rhos 0.0 0.05 0.1 0.2")
    parser.add_argument("--sigma", type=float, default=1.0, help="base std σ in [0,1] space (보통 1.0)")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--suffix", type=str, default="{rho}", help="파일명 접미사 템플릿. 예: '{rho}' -> _rho0.10")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if args.rho is None and not args.rhos:
        raise ValueError("하나 이상 지정: --rho 또는 --rhos")

    rho_list = []
    if args.rho is not None:
        rho_list.append(args.rho)
    if args.rhos:
        rho_list.extend(args.rhos)
    rho_list = sorted(set(rho_list))

    if in_path.is_file():
        out_path.mkdir(parents=True, exist_ok=True)
        img = load_image(in_path)
        for rho in rho_list:
            noisy = add_gaussian_noise(img, rho=rho, sigma=args.sigma, seed=args.seed)
            name = in_path.stem + f"_rho{rho:.2f}" + in_path.suffix
            save_image(noisy, out_path / name)
    else:
        # directory
        out_path.mkdir(parents=True, exist_ok=True)
        img_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
            img_files.extend(in_path.rglob(ext))
        img_files = sorted(img_files)

        for p in tqdm(img_files, desc="noising"):
            img = load_image(p)
            for rho in rho_list:
                noisy = add_gaussian_noise(img, rho=rho, sigma=args.sigma, seed=args.seed)
                # mirror relative path
                rel = p.relative_to(in_path)
                subdir = out_path / f"rho_{rho:.2f}" / rel.parent
                subdir.mkdir(parents=True, exist_ok=True)
                out_file = subdir / (p.stem + p.suffix)
                save_image(noisy, out_file)

if __name__ == "__main__":
    main()