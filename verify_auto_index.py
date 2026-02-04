import os
import cv2
import numpy as np
import argparse
from math import ceil, sqrt
from scipy.ndimage import laplace

def parse_args():
    p = argparse.ArgumentParser(description="Auto verify stitching for index‐named tiles")
    p.add_argument("--tiles_dir",   required=True, help="Folder with tiles <slideID>_<idx>.png/.tif")
    p.add_argument("--stitched",    required=True, help="Path to stitched_result.png")
    p.add_argument("--overlap",     type=float, default=0.10, help="Fractional overlap")
    return p.parse_args()

def find_tiles_by_index(input_dir):
    """Detect slideID and build a grid of filenames by index."""
    exts = (".png", ".tif", ".jpg")
    entries = []
    for f in os.listdir(input_dir):
        if not f.lower().endswith(exts): continue
        name, _ = os.path.splitext(f)
        parts = name.rsplit("_",1)
        if len(parts)!=2 or not parts[1].isdigit(): continue
        slide, idx = parts[0], int(parts[1])
        entries.append((slide, idx, f))
    if not entries:
        raise RuntimeError("No tiles found in "+input_dir)
    slideID = entries[0][0]
    files = {idx:f for s,idx,f in entries if s==slideID}
    N = len(files)
    # infer grid dims as near-square
    C = int(ceil(sqrt(N)))
    R = int(ceil(N/C))
    grid = [[None]*C for _ in range(R)]
    for idx,f in files.items():
        r,c = divmod(idx, C)
        if r<R and c<C:
            grid[r][c] = f
    # check coverage of first N positions
    missing = sum(1 for r in range(R) for c in range(C) if grid[r][c] is None)
    if missing > (R*C - N):
        raise RuntimeError(f"Missing tiles: grid {R}×{C}, but {N} files")
    return slideID, R, C, grid

def load_grid(input_dir, grid, W, H):
    R,C = len(grid), len(grid[0])
    arr = np.zeros((R,C,H,W), dtype=np.float32)
    for i in range(R):
        for j in range(C):
            fname = grid[i][j]
            if fname is None:
                raise RuntimeError(f"Missing tile at {i},{j}")
            img = cv2.imread(os.path.join(input_dir,fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Cannot read "+fname)
            if img.shape[::-1] != (W,H):
                img = cv2.resize(img,(W,H))
            arr[i,j] = img.astype(np.float32)
    return arr

def compute_overlap_metrics(grid, overlap):
    R,C,H,W = grid.shape
    sh = int(round(H*(1-overlap)))
    sw = int(round(W*(1-overlap)))
    diffs=[]
    for i in range(R):
        for j in range(C):
            base = grid[i,j]
            if j<C-1:
                right = grid[i,j+1]
                diffs.append((base[:,-sw:] - right[:,:sw]).ravel())
            if i<R-1:
                bot = grid[i+1,j]
                diffs.append((base[-sh:,:] - bot[:sh,:]).ravel())
    a = np.concatenate(diffs)
    return np.mean(np.abs(a)), np.std(a), np.sum(a*a)

def split_stitched(stitched, R, C, W, H, overlap):
    """Extract tiles from the stitched mosaic using nominal translations."""
    sh = int(round(H*(1-overlap)))
    sw = int(round(W*(1-overlap)))
    tiles = np.zeros((R,C,H,W), dtype=np.float32)
    for i in range(R):
        for j in range(C):
            y,x = i*sh, j*sw
            tiles[i,j] = stitched[y:y+H, x:x+W].astype(np.float32)
    return tiles

def main():
    args = parse_args()
    # detect tiles
    slideID, R, C, grid = find_tiles_by_index(args.tiles_dir)
    print(f"Detected slideID={slideID}, grid={R}×{C}")
    # detect native tile size
    sample = cv2.imread(os.path.join(args.tiles_dir, grid[0][0]), cv2.IMREAD_GRAYSCALE)
    H0, W0 = sample.shape
    print(f"Tile size: {W0}×{H0}")
    # load raw grid
    grid_raw = load_grid(args.tiles_dir, grid, W0, H0)
    print("Computing pre-correction overlap metrics...")
    pre_mean, pre_std, pre_ssd = compute_overlap_metrics(grid_raw, args.overlap)
    print(f"Pre-correction: mean_abs_diff={pre_mean:.2f}, std={pre_std:.2f}, SSD={pre_ssd:.2e}")
    # load stitched mosaic
    stitched = cv2.imread(args.stitched, cv2.IMREAD_GRAYSCALE)
    if stitched is None:
        raise RuntimeError("Cannot read stitched image: "+args.stitched)
    # split into tiles
    grid_post = split_stitched(stitched, R, C, W0, H0, args.overlap)
    print("Computing post-correction overlap metrics...")
    post_mean, post_std, post_ssd = compute_overlap_metrics(grid_post, args.overlap)
    print(f"Post-correction: mean_abs_diff={post_mean:.2f}, std={post_std:.2f}, SSD={post_ssd:.2e}")
    # laplacian variance (sharpness) of stitched mosaic
    lap = laplace(stitched.astype(np.float32))
    print(f"Laplacian variance (stitched): {np.var(lap):.2f}")

if __name__=="__main__":
    main()
