# streamlit_app.py
import cv2
import zipfile
import tempfile
import numpy as np
import torch
import os
from collections import Counter
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFile

# --- Model Definition ---
# This assumes 'models/stitchnet.py' exists and is correct.
try:
    from models.stitchnet import StitchNet
except ImportError:
    st.error("Could not import StitchNet. Make sure 'models/stitchnet.py' is in your project directory.")
    st.stop()

# --- Global Safeguards ---
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- UI Configuration ---
st.set_page_config(page_title="PANDA Tile Stitcher", page_icon="🧬", layout="wide")
st.sidebar.title("⚙️ Stitching Options")

# --- User Inputs in Sidebar ---
weights_up = st.sidebar.file_uploader("Upload Model Weights (.pth)", ["pth"])
tile_px = st.sidebar.slider("Processing Tile Size (px)", 128, 1024, 512, 64)

# --- MANUAL GRID OVERRIDE ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Manual Grid Control")
st.sidebar.info("If auto-detection fails, manually set the grid width (columns) here.")
manual_cols = st.sidebar.number_input("Grid Width (Columns)", min_value=0, value=0,
                                      help="Set to 0 for auto-detection. A common value for PANDA is 35 or 36.")

# --- Handle Model Weights ---
weights_dir = Path(__file__).parent / "weights"
weights_dir.mkdir(exist_ok=True)
default_weights = "models/stitchnet_v2_ep10.pth"  # Uses the correctly trained model
w_path = default_weights

if weights_up:
    w_path = weights_dir / "custom_weights.pth"
    w_path.write_bytes(weights_up.getbuffer())
    st.sidebar.success(f"Using uploaded weights: {weights_up.name}")
elif not os.path.exists(default_weights):
    st.sidebar.warning(f"Default weights '{default_weights}' not found. Please upload a model.")
    w_path = None
else:
    st.sidebar.info(f"Using default weights.")

# --- Main Application Area ---
st.markdown("<h2 style='text-align:center'>🔬 High-Resolution Tile Stitcher</h2>", unsafe_allow_html=True)
uploads = st.file_uploader("Drop PNG tiles or a single ZIP archive", ["png", "zip"], accept_multiple_files=True)


# --- Helper Functions ---
def infer_grid_robust(indices: list[int]) -> tuple[int, int]:
    """
    Infers grid dimensions by finding the mode of consecutive run lengths.
    This is the most reliable heuristic for grid width.
    """
    if len(indices) < 2: return 1, len(indices)

    indices.sort()
    runs = []
    current_run = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    if not runs: return 1, len(indices)

    cols = Counter(runs).most_common(1)[0][0]
    rows = -(-max(indices) // cols) if cols > 0 else 0
    return rows, cols


def find_largest_inner_rectangle(image_mask):
    """
    Finds the largest content-filled rectangle to crop away black borders.
    """
    contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0, image_mask.shape[1], image_mask.shape[0]

    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


# --- Stitching Logic ---
if st.button("Stitch Tiles 🧬") and uploads:
    if not w_path or not os.path.exists(w_path):
        st.error("Model weights are not available. Please upload a .pth file.");
        st.stop()

    with st.spinner("Preparing tiles..."):
        # Unpack and organize tile paths
        tmpdir = Path(tempfile.mkdtemp())
        paths = []
        for f in uploads:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    z.extractall(tmpdir)
                    paths.extend(tmpdir / n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".tif")))
            else:
                (tmpdir / f.name).write_bytes(f.getbuffer());
                paths.append(tmpdir / f.name)

        if len(paths) < 2: st.error("Stitching requires at least 2 tiles."); st.stop()

        paths.sort(key=lambda p: int(p.stem.split('_')[-1]))
        indices = [int(p.stem.split('_')[-1]) for p in paths]
        path_dict = {idx: str(p) for idx, p in zip(indices, paths)}

    with st.spinner("Analyzing grid and loading model..."):
        # Use manual grid width if provided, otherwise auto-detect.
        if manual_cols > 0:
            cols = manual_cols
            rows = -(-max(indices) // cols) if cols > 0 else 0
            st.success(f"Using manual grid dimensions: {rows} × {cols}")
        else:
            rows, cols = infer_grid_robust(indices)
            st.success(f"Auto-detected grid dimensions: {rows} × {cols}")

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = StitchNet().to(dev).eval();
        net.requires_grad_(False)
        net.load_state_dict(torch.load(w_path, map_location=dev))

    H, W = tile_px, tile_px;
    pad = int(0.20 * W)  # Increased padding for more aggressive warps
    canvas = np.zeros((rows * H + 2 * pad, cols * W + 2 * pad, 3), np.float32)
    alpha_mask = np.zeros(canvas.shape[:2], np.float32)

    bar = st.progress(0, "Stitching tiles...")
    min_idx = indices[0] if indices else 0
    for i, idx in enumerate(indices):
        r, c = divmod(idx - min_idx, cols)

        current_img = cv2.resize(cv2.imread(path_dict[idx]), (W, H))
        current_tensor = torch.from_numpy(current_img.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(dev)

        ref_tensor = None
        # Use original neighbor image as stable reference.
        if c > 0 and (idx - 1) in path_dict:
            ref_img = cv2.resize(cv2.imread(path_dict[idx - 1]), (W, H))
            ref_tensor = torch.from_numpy(ref_img.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(dev)
        elif r > 0 and (idx - cols) in path_dict:
            ref_img = cv2.resize(cv2.imread(path_dict[idx - cols]), (W, H))
            ref_tensor = torch.from_numpy(ref_img.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(dev)

        padded_cur = torch.nn.functional.pad(current_tensor, (pad, pad, pad, pad), mode='replicate')

        if ref_tensor is not None:
            padded_ref = torch.nn.functional.pad(ref_tensor, (pad, pad, pad, pad), mode='replicate')
            with torch.no_grad():
                warped, _ = net(padded_ref, padded_cur)
        else:
            warped = padded_cur

        tile_to_paste = warped.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        y, x = r * H, c * W

        # Feather edges for smoother blending
        feather = np.ones((H + 2 * pad, W + 2 * pad, 1), dtype=np.float32)
        border = int(pad * 0.5)
        if border > 0:
            feather[:border, :] *= np.linspace(0, 1, border)[:, None, None];
            feather[-border:, :] *= np.linspace(1, 0, border)[:, None, None]
            feather[:, :border] *= np.linspace(0, 1, border)[None, :, None];
            feather[:, -border:] *= np.linspace(1, 0, border)[None, :, None]

        canvas[y:y + H + 2 * pad, x:x + W + 2 * pad] += tile_to_paste * feather
        alpha_mask[y:y + H + 2 * pad, x:x + W + 2 * pad] += feather.squeeze()
        bar.progress((i + 1) / len(indices))

    with st.spinner("Finalizing image..."):
        alpha_mask[alpha_mask == 0] = 1e-6
        stitched_float = (canvas / alpha_mask[..., None]).clip(0, 1)
        stitched_padded = (stitched_float * 255).astype(np.uint8)

        # Auto-crop black borders
        gray_mask = cv2.cvtColor(stitched_padded, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        x_crop, y_crop, w_crop, h_crop = find_largest_inner_rectangle(binary_mask)
        final_stitched = stitched_padded[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]

    if final_stitched.size == 0:
        st.error(
            "Stitching resulted in an empty image. The grid dimensions may be incorrect. Try setting the width manually.")
    else:
        h_final, w_final = final_stitched.shape[:2]
        max_side = 4000
        preview = cv2.resize(final_stitched, (int(w_final * max_side / max(h_final, w_final)),
                                              int(h_final * max_side / max(h_final, w_final))),
                             interpolation=cv2.INTER_AREA) if max(h_final, w_final) > max_side else final_stitched

        st.subheader("🖼️ Stitched Panorama (Preview)")
        st.image(preview, channels="BGR", use_container_width=True, caption=f"Final dimensions: {h_final}×{w_final}px")

        success, encoded_image = cv2.imencode(".png", final_stitched)
        if success:
            st.download_button(
                "💾 Download Full-Resolution PNG",
                data=encoded_image.tobytes(),
                file_name="stitched_panorama.png",
                mime="image/png"
            )
