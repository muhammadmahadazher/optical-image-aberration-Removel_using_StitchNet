import os
import argparse
import numpy as np
import cv2
from math import ceil, sqrt
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

def parse_args():
    p = argparse.ArgumentParser(description="Auto stitch by index")
    p.add_argument("--input_dir", required=True,
                   help="Folder of tiles named <slideID>_<idx>.png/.tif/.jpg")
    p.add_argument("--overlap", type=float, default=0.10,
                   help="Fractional overlap (default 0.10)")
    return p.parse_args()

def find_tiles_by_index(input_dir):
    exts = (".png", ".tif", ".jpg")
    entries = []
    for f in os.listdir(input_dir):
        if not f.lower().endswith(exts): continue
        name, _ = os.path.splitext(f)
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit(): continue
        slideID, idx = parts[0], int(parts[1])
        entries.append((slideID, idx, f))
    if not entries:
        raise RuntimeError("No tiles found in "+input_dir)
    # assume single slideID
    slideID = entries[0][0]
    files = {idx: fname for sid,idx,fname in entries if sid==slideID}
    N = len(files)
    # build grid dims
    cols = int(ceil(sqrt(N)))
    rows = int(ceil(N/cols))
    # assign index->(row,col)
    grid = [[None]*cols for _ in range(rows)]
    for idx, fname in files.items():
        r, c = divmod(idx, cols)
        if r<rows and c<cols:
            grid[r][c] = fname
    # verify coverage for first rows*cols>=N positions
    missing = [i for row in grid for i in row if i is None]
    if len(missing) > (rows*cols - N):
        raise RuntimeError("Missing tiles or unexpected indices")
    return slideID, rows, cols, grid

def load_grid(input_dir, grid, W, H):
    R, C = len(grid), len(grid[0])
    arr = np.zeros((R, C, H, W), dtype=np.float32)
    for i in range(R):
        for j in range(C):
            fname = grid[i][j]
            if fname is None:
                raise RuntimeError(f"Missing tile at {i},{j}")
            img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img.shape[::-1] != (W,H):
                img = cv2.resize(img, (W,H))
            arr[i,j] = img.astype(np.float32)
    return arr

def distortion_field(X, Y, d, x_c, y_c, l):
    xt, yt = (X-x_c)/l, (Y-y_c)/l
    dx = (d[0]*xt*yt + d[1]*xt**2 + d[2]*yt**2 +
          d[3]*xt**2*yt + d[4]*xt*yt**2 + d[5]*xt**3 + d[6]*yt**3)
    dy = (d[7]*xt*yt + d[8]*xt**2 + d[9]*yt**2 +
          d[3]*xt**2*yt + d[4]*xt*yt**2 + d[5]*xt**3 + d[6]*yt**3)
    return dx, dy

def warp_image(img, d, x_c, y_c, l):
    H, W = img.shape
    Y, X = np.mgrid[0:H, 0:W]
    dx, dy = distortion_field(X, Y, d, x_c, y_c, l)
    coords = np.stack([Y+dy, X+dx], axis=0)
    return map_coordinates(img, coords, order=3, mode='reflect')

def overlap_error(d, grid, overlap):
    R,C,H,W = grid.shape
    sh, sw = int(round(H*(1-overlap))), int(round(W*(1-overlap)))
    x_c, y_c, l = W//2, H//2, max(W,H)
    errs = []
    for i in range(R):
        for j in range(C):
            base = warp_image(grid[i,j], d, x_c, y_c, l)
            if j<C-1:
                right = warp_image(grid[i,j+1], d, x_c, y_c, l)
                errs.append((base[:,-sw:]-right[:,:sw]).ravel())
            if i<R-1:
                bot = warp_image(grid[i+1,j], d, x_c, y_c, l)
                errs.append((base[-sh:,:]-bot[:sh,:]).ravel())
    return np.concatenate(errs)

def blend_full(grid, d, overlap):
    R,C,H,W = grid.shape
    sh, sw = int(round(H*(1-overlap))), int(round(W*(1-overlap)))
    Hf, Wf = sh*(R-1)+H, sw*(C-1)+W
    canvas = np.zeros((Hf,Wf), dtype=np.float32)
    weight = np.zeros_like(canvas)
    x_c, y_c, l = W//2, H//2, max(W,H)
    for i in range(R):
        for j in range(C):
            y, x = i*sh, j*sw
            warped = warp_image(grid[i,j], d, x_c, y_c, l)
            canvas[y:y+H, x:x+W] += warped
            weight[y:y+H, x:x+W] += 1
    canvas[weight>0] /= weight[weight>0]
    return np.clip(canvas,0,255).astype(np.uint8)

def main():
    args = parse_args()
    sid, R, C, grid = find_tiles_by_index(args.input_dir)
    print(f"🔬 SlideID={sid}, grid={R}×{C}")
    # detect tile size
    sample = cv2.imread(os.path.join(args.input_dir, grid[0][0]), cv2.IMREAD_GRAYSCALE)
    H0, W0 = sample.shape
    print(f"📏 Tile size: {W0}×{H0}")
    # calibration subset
    subR, subC = min(3,R), min(3,C)
    sub = [[grid[i][j] for j in range(subC)] for i in range(subR)]
    grid_sub = load_grid(args.input_dir, sub, W0, H0)
    print(f"🧪 Calibrating on {subR}×{subC} subset...")
    d0 = np.zeros(10)
    res = least_squares(overlap_error, d0, args=(grid_sub, args.overlap),
                        verbose=2, max_nfev=20)
    d_opt = res.x
    print(f"✅ Distortion params optimized: {d_opt}")
    full = load_grid(args.input_dir, grid, W0, H0)
    print("🎨 Blending full mosaic...")
    out = blend_full(full, d_opt, args.overlap)
    cv2.imwrite("stitched_result.png", out)
    print("✨ Saved: stitched_result.png")

if __name__=="__main__":
    main()
