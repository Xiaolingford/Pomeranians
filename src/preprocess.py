import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ----- utils -----
def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def gray_world_white_balance(img_rgb):
    img = img_rgb.astype(np.float32) + 1e-6
    mean = img.reshape(-1, 3).mean(axis=0)
    gain = np.clip(mean.mean() / mean, 0.95, 1.05)  # limit to ±5%
    wb = np.clip(img * gain, 0, 255).astype(np.uint8)
    return wb

def illumination_flatten(img_rgb, sigma=21, alpha=-0.005, bias=100):
    """
    Extremely soft illumination flattening (tiny whitening effect).
    """
    bg = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.addWeighted(img_rgb, 1.0, bg, alpha, bias)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    return flat


def clahe_on_v(img_rgb, clip=0.5, tile=8):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def denoise_soft(
    img_rgb,
    method="nlmeans",
    h=3.0,                # luminance filter strength (higher = smoother)
    hColor=5.0,           # color filter strength
    templateWindowSize=7,
    searchWindowSize=21,
    bilateral_d=5,
    bilateral_sigmaColor=25.0,
    bilateral_sigmaSpace=5.0,
):
    """
    Balanced denoising (stronger than microscopic, lighter than defaults).
    """
    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(
            img_rgb, None, h, hColor, templateWindowSize, searchWindowSize
        )
    elif method == "bilateral":
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        den = cv2.bilateralFilter(bgr, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
        return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
    else:
        return img_rgb.copy()


def contrast_stretch(img_rgb, low=5, high=95):
    prc1 = np.percentile(img_rgb, low)
    prc2 = np.percentile(img_rgb, high)
    if prc2 - prc1 < 1e-3:  # avoid divide by near-zero
        return img_rgb.copy()
    out = np.clip((img_rgb - prc1) * (255.0 / (prc2 - prc1 + 1e-6)), 0, 255)
    return out.astype(np.uint8)

def resize_image(img_rgb, size=224):
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

def process_one(
    img_rgb,
    size=224,
    apply_wb=True,
    apply_flatten=False,
    apply_clahe=True,
    apply_denoise=True,
    apply_stretch=True,
    flatten_alpha=-0.01,
    flatten_bias=105,
    clahe_clip=0.5,
    stretch_low=5,
    stretch_high=95,
):
    x = resize_image(img_rgb, size=size)

    if apply_wb:
        x = gray_world_white_balance(x)

    if apply_flatten:
        x = illumination_flatten(x, sigma=21, alpha=flatten_alpha, bias=flatten_bias)

    if apply_clahe:
        x = clahe_on_v(x, clip=clahe_clip, tile=8)

    if apply_denoise:
        x = denoise_soft(x, h=5.0, hColor=7.0)


    if apply_stretch:
        x = contrast_stretch(x, low=stretch_low, high=stretch_high)

    return x

def main():
    ap = argparse.ArgumentParser("Batch preprocessing for classification or detection")
    ap.add_argument("--task", choices=["classification", "detection"], default="classification")
    ap.add_argument("--input", required=True, help="Input folder (with images or YOLO structure)")
    ap.add_argument("--outdir", default="data_preprocessed", help="Output folder")
    ap.add_argument("--keep-structure", action="store_true", help="Preserve subfolders (classification only)")
    ap.add_argument("--size", type=int, default=224, help="Target size (width=height)")

    # New tuning args
    ap.add_argument("--no-wb", action="store_true", help="Disable white balance")
    ap.add_argument("--no-flatten", action="store_true", help="Disable illumination flattening")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    ap.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    ap.add_argument("--no-stretch", action="store_true", help="Disable contrast stretch")

    ap.add_argument("--flatten-alpha", type=float, default=-0.05, help="Flatten blending alpha")
    ap.add_argument("--flatten-bias", type=int, default=120, help="Flatten blending bias")
    ap.add_argument("--clahe-clip", type=float, default=0.3, help="CLAHE clip limit")
    ap.add_argument("--stretch-low", type=int, default=1, help="Low percentile for contrast stretch")
    ap.add_argument("--stretch-high", type=int, default=99, help="High percentile for contrast stretch")

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    def run_process(src, dst):
        rgb = read_rgb(src)
        out = process_one(
            rgb,
            size=args.size,
            apply_wb=not args.no_wb,
            apply_flatten=not args.no_flatten,
            apply_clahe=not args.no_clahe,
            apply_denoise=not args.no_denoise,
            apply_stretch=not args.no_stretch,
            flatten_alpha=args.flatten_alpha,
            flatten_bias=args.flatten_bias,
            clahe_clip=args.clahe_clip,
            stretch_low=args.stretch_low,
            stretch_high=args.stretch_high,
        )
        cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    if args.task == "classification":
        items = []
        if args.keep_structure:
            subdirs = [d for d in in_dir.iterdir() if d.is_dir()]
            for d in subdirs:
                dst = out_dir / d.name
                dst.mkdir(parents=True, exist_ok=True)
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
                        items.append((p, dst/p.name))
        else:
            for p in sorted(in_dir.iterdir()):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
                    items.append((p, out_dir/p.name))

        ok = 0
        for src, dst in tqdm(items, desc="preprocess", unit="img"):
            try:
                run_process(src, dst)
                ok += 1
            except Exception as e:
                print(f"[WARN] {src.name}: {e}")
        print(f"Done. {ok}/{len(items)} images written to {out_dir}")

    elif args.task == "detection":
        # Check if YOLO-style subfolders exist
        img_in = in_dir / "images" / "train"
        lbl_in = in_dir / "labels" / "train"


        if img_in.exists() and lbl_in.exists():
            # Standard YOLO structure
            items = [p for p in img_in.iterdir() if p.suffix.lower() in 
                     {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}]
        else:
            # Flat structure: images and labels mixed in root
            img_in = in_dir
            lbl_in = in_dir
            items = [p for p in in_dir.iterdir() if p.suffix.lower() in 
                     {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}]

        # Prepare output dirs
        img_out = out_dir / "images"; img_out.mkdir(parents=True, exist_ok=True)
        lbl_out = out_dir / "labels"; lbl_out.mkdir(parents=True, exist_ok=True)

        ok = 0
        for src in tqdm(items, desc="preprocess", unit="img"):
            try:
                # Process image
                run_process(src, img_out/src.name)

                # Copy label if exists
                lbl_src = lbl_in / (src.stem + ".txt")
                lbl_dst = lbl_out / (src.stem + ".txt")
                if lbl_src.exists():
                    with open(lbl_src) as f:
                        lines = f.readlines()
                    with open(lbl_dst, "w") as f:
                        for line in lines:
                            f.write(line)

                ok += 1
            except Exception as e:
                print(f"[WARN] {src.name}: {e}")
        print(f"Done. {ok}/{len(items)} images+labels written to {out_dir}")


if __name__ == "__main__":
    main()
