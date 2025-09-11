from pathlib import Path
import argparse
from tqdm import tqdm
from dataset import letterbox_pad, IM_SIZE, _read_rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", type=str, default="./data", help="input root with algae/ and microplastics/")
    ap.add_argument("--outroot", type=str, default="./data_resized", help="output root")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    inroot = (ROOT / args.inroot).resolve()
    outroot = (ROOT / args.outroot).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    for cname in ["algae", "microplastics"]:
        src = inroot / cname
        dst = outroot / cname
        dst.mkdir(parents=True, exist_ok=True)

        files = sorted([*src.glob("*.jpg"), *src.glob("*.jpeg"), *src.glob("*.png")])
        bad = 0
        for p in tqdm(files, desc=cname):
            try:
                img = _read_rgb(p)
            except Exception:
                bad += 1
                continue
            lb = letterbox_pad(img, IM_SIZE)
            (dst / p.name).parent.mkdir(parents=True, exist_ok=True)
            # use PIL to save to preserve Unicode
            from PIL import Image
            Image.fromarray(lb).save(dst / p.name)
        if bad:
            print(f"[WARN] {cname}: skipped {bad} unreadable image(s)")

    print(f"Saved to {outroot}")

if __name__ == "__main__":
    main()
