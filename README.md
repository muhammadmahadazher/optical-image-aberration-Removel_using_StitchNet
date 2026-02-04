# optical-image-aberration-Removel_using_StitchNet
Optical aberration Removel using AI High resolution and large field of view imaging using a stitching procedure coupled with distortion corrections

Optical Aberration Removal using AI
Quick Start Guide
This guide provides commands to set up the environment, run the AI-based Streamlit app, train the AI model, and run a classical stitching script for comparison.

1. Setup Environment
bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
2. Run Streamlit App (AI Stitching)
bash
# Ensure model weights are present (e.g., 'weights/stitchnet_v2_ep10.pth')
# Or upload them via the app's sidebar.

# Launch the Streamlit web application
streamlit run streamlit_app.py

# In the app: Upload tiles, use auto-detect grid, or manually set "Grid Width (Columns)"
# (e.g., 35 or 36 for PANDA slides if auto-detect fails).
# Click "Stitch Tiles 🧬".
3. Train the AI Model
bash
# Ensure your PANDA tiles are in data/panda_tiles/ (e.g., data/panda_tiles/slide_id_001/*.png)

# Start training the StitchNet model
# --workers 0 is recommended for stability on Windows
python train_stitch_v2.py --tiles_dir data/panda_tiles --epochs 10 --batch_size 8 --lr 1e-4 --workers 0
4. Run Classical Stitching Script (Non-AI Comparison)
bash
# Specify the folder of a single slide to stitch
# Replace <YOUR_SLIDE_ID_FOLDER> with your actual slide folder name (e.g., 00a7fb880dc12c5de82df39b30533da9)
python stitch_auto.py --input_dir data/panda_tiles/<YOUR_SLIDE_ID_FOLDER>

# Output will be 'stitched_result.png' in your project root.
