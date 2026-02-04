import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

# ========== PARAMETERS ==========
input_dir = "synthetic_tiles"   # Folder with tiles named tile_{i}_{j}.png
grid_shape = (3, 3)             # Change as needed
img_shape = (256, 256)          # Size of each tile (pixels)
overlap = 0.10                  # Fractional overlap (e.g., 0.10 for 10%)
distortion_order = 3            # Cubic polynomial, as in the paper
max_iter = 30                   # Number of optimization iterations

# ========== DISTORTION MODEL ==========
def distortion_field(x, y, d, x_c, y_c, l):
    # Implements Eq. 9 from the paper (non-affine cubic polynomial)
    xt = (x - x_c) / l
    yt = (y - y_c) / l
    # Only non-affine terms (see paper, Table 1 and Eq. 9)
    # Modes: [xt*yt, xt**2, yt**2, xt**2*yt, xt*yt**2, xt**3, yt**3]
    dx = (d[0]*xt*yt + d[1]*xt**2 + d[2]*yt**2 +
          d[3]*xt**2*yt + d[4]*xt*yt**2 + d[5]*xt**3 + d[6]*yt**3)
    dy = (d[7]*xt*yt + d[8]*xt**2 + d[9]*yt**2 +
          d[3]*xt**2*yt + d[4]*xt*yt**2 + d[5]*xt**3 + d[6]*yt**3)
    return dx, dy

def warp_image(img, d, x_c, y_c, l):
    h, w = img.shape
    Y, X = np.mgrid[0:h, 0:w]
    dx, dy = distortion_field(X, Y, d, x_c, y_c, l)
    coords = np.array([Y + dy, X + dx])
    return map_coordinates(img, coords, order=3, mode='reflect')

# ========== LOAD IMAGES ==========
def load_grid_images(input_dir, grid_shape, img_shape):
    grid = []
    for i in range(grid_shape[0]):
        row = []
        for j in range(grid_shape[1]):
            fname = os.path.join(input_dir, f'tile_{i}_{j}.png')
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing image: {fname}")
            img = cv2.resize(img, img_shape)
            row.append(img.astype(np.float32))
        grid.append(row)
    return np.array(grid)

# ========== OPTIMIZATION COST FUNCTION ==========
def overlap_error(params, grid, grid_shape, img_shape, overlap):
    d = params[:10]
    t = params[10:].reshape(grid_shape[0], grid_shape[1], 2)
    h, w = img_shape
    step_h = int(round(h * (1 - overlap)))
    step_w = int(round(w * (1 - overlap)))
    l = max(h, w)
    x_c, y_c = w // 2, h // 2
    error = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img = grid[i, j]
            tx, ty = t[i, j]
            img_warp = warp_image(img, d, x_c, y_c, l)
            # Horizontal overlap
            if j < grid_shape[1] - 1:
                img_r = grid[i, j+1]
                tx_r, ty_r = t[i, j+1]
                img_r_warp = warp_image(img_r, d, x_c, y_c, l)
                left = img_warp[:, -step_w:]
                right = img_r_warp[:, :step_w]
                error.append((left - right).ravel())
            # Vertical overlap
            if i < grid_shape[0] - 1:
                img_b = grid[i+1, j]
                tx_b, ty_b = t[i+1, j]
                img_b_warp = warp_image(img_b, d, x_c, y_c, l)
                top = img_warp[-step_h:, :]
                bottom = img_b_warp[:step_h, :]
                error.append((top - bottom).ravel())
    return np.concatenate(error)

# ========== BLENDING ==========
def blend_grid(grid, d, t, grid_shape, img_shape, overlap):
    h, w = img_shape
    step_h = int(round(h * (1 - overlap)))
    step_w = int(round(w * (1 - overlap)))
    stitched_h = step_h * (grid_shape[0] - 1) + h
    stitched_w = step_w * (grid_shape[1] - 1) + w
    stitched = np.zeros((stitched_h, stitched_w), dtype=np.float32)
    weight = np.zeros_like(stitched)
    l = max(h, w)
    x_c, y_c = w // 2, h // 2
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            y = i * step_h
            x = j * step_w
            img = grid[i, j]
            img_warp = warp_image(img, d, x_c, y_c, l)
            stitched[y:y+h, x:x+w] += img_warp
            weight[y:y+h, x:x+w] += 1
    stitched[weight > 0] /= weight[weight > 0]
    return np.clip(stitched, 0, 255).astype(np.uint8)

# ========== MAIN WORKFLOW ==========
if __name__ == "__main__":
    # Load images
    grid = load_grid_images(input_dir, grid_shape, img_shape)

    # Initial guess: no distortion, regular grid
    d0 = np.zeros(10)
    t0 = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            t0.extend([j * img_shape[1] * (1 - overlap), i * img_shape[0] * (1 - overlap)])
    params0 = np.concatenate([d0, t0])

    # Optimize (Gauss-Newton, as in paper)
    res = least_squares(
        overlap_error, params0,
        args=(grid, grid_shape, img_shape, overlap),
        verbose=2, max_nfev=max_iter
    )
    d_opt = res.x[:10]
    t_opt = res.x[10:].reshape(grid_shape[0], grid_shape[1], 2)

    # Blend and save
    stitched = blend_grid(grid, d_opt, t_opt, grid_shape, img_shape, overlap)
    cv2.imwrite("stitched_result.png", stitched)
    print("Distortion-corrected stitching complete. Output saved as stitched_result.png")
