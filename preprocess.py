import os
import argparse
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess PANDA Tiles: resize, normalize, binarize masks")
    parser.add_argument("--train_dir", required=True,
                        help="Path to data/train folder with images")
    parser.add_argument("--mask_dir", required=True,
                        help="Path to data/masks folder with masks")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where processed data will be saved")
    parser.add_argument("--size", nargs=2, type=int, metavar=('W','H'),
                        default=(256,256),
                        help="Resize width and height (default: 256 256)")
    return parser.parse_args()

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def preprocess(train_dir, mask_dir, output_dir, size):
    out_img = os.path.join(output_dir, "images")
    out_mask = os.path.join(output_dir, "masks")
    ensure_dirs([out_img, out_mask])

    img_files = sorted(f for f in os.listdir(train_dir)
                       if f.lower().endswith(('.png','.jpg','.tif')))
    for fname in img_files:
        base, _ = os.path.splitext(fname)
        # load and resize image
        img_path = os.path.join(train_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0

        # save normalized image as PNG
        out_img_path = os.path.join(out_img, base + ".png")
        scaled = (img_arr * 255).astype(np.uint8)
        Image.fromarray(scaled).save(out_img_path)

        # load corresponding mask if it exists
        mask_fname = base + ".png"
        mask_path = os.path.join(mask_dir, mask_fname)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(size, Image.NEAREST)
            mask_arr = np.asarray(mask, dtype=np.uint8)
            # binarize: 0 background, 1 foreground
            mask_bin = (mask_arr > 127).astype(np.uint8) * 255
            out_mask_path = os.path.join(out_mask, base + ".png")
            Image.fromarray(mask_bin).save(out_mask_path)
        else:
            print(f"Warning: no mask found for {fname}")

if __name__ == "__main__":
    args = parse_args()
    preprocess(args.train_dir, args.mask_dir, args.output_dir, tuple(args.size))
    print("Preprocessing complete.")
