# 🔬 StitchNet: Optical Image Aberration Removal & Seamless Stitching 🧬

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**StitchNet** is an advanced deep learning framework designed to solve the "last mile" of microscopy imaging: **seamless tile stitching**. By combining ResNet-based flow prediction with classical geometric optimization, it removes optical aberrations and aligns high-resolution tiles (like those in the PANDA dataset) into perfect, artifact-free panoramas. 🚀

---

## 🎨 System Architecture & Workflow

### 🛠️ High-Level Pipeline
```text
      ┌─────────────────────────────┐
      │  📥 Input: Raw Image Tiles  │
      └──────────────┬──────────────┘
                     │
      🔍 [Step 1: Grid Inference] ──────────────────┐
         Identifies Rows x Cols from Filenames      │
                     │                              │
      🧠 [Step 2: Core Processing Engine]           │
         ┌───────────┴───────────┐                  │
         ▼                       ▼                  │
  🌈 AI-Powered Mode      🔢 Classical Mode         │
  (StitchNet Model)       (Poly-Optimization)       │
  Predicts pixel-wise     Fits Cubic Fields to      │
  warp/flow fields.       overlap regions.          │
         │                       │                  │
         └───────────┬───────────┘                  │
                     ▼                              │
      ✨ [Step 3: Warping & Blending] <─────────────┘
         Applies transforms + Linear Feathering
                     │
      ✂️ [Step 4: Smart Auto-Crop]
         Finds Largest Inner Content Rectangle
                     │
      ┌──────────────┴──────────────┐
      │  🏁 Output: Seamless Slide  │
      └─────────────────────────────┘
```

### 🧬 Inside the "StitchNet" Model
```text
  Tile Pair (Ref, Nbr)
          │
    [ResNet18 Encoder] ───┐
          │               │
    [Feature Matching] ───┼──> [Flow Predictor]
          │               │           │
    [Warping Layer] <─────┘           ▼
          │                   [Pixel Warp Map]
          ▼
    Aligned Result
```

---

## ✨ Key Features

*   **🤖 Deep Learning Precision:** Uses a custom `StitchNet` architecture to handle non-linear optical distortions.
*   **🧪 Hybrid Approach:** Switch between AI prediction and classical Gauss-Newton optimization.
*   **🌐 Modern Web Interface:** A fully-featured **Streamlit** dashboard for drag-and-drop stitching.
*   **📐 Intelligent Auto-Grid:** Detects your slide layout automatically without manual input.
*   **✂️ Clean Finishes:** Automatic removal of black "stitching borders" using content-aware cropping.
*   **⚡ Optimized Performance:** Multi-threaded tile loading and GPU-accelerated warping.

---

## 🚀 Getting Started

### 1️⃣ Clone the Laboratory
```bash
git clone https://github.com/muhammadmahadazher/optical-image-aberration-Removel_using_StitchNet.git
cd optical-image-aberration-Removel_using_StitchNet
```

### 2️⃣ Prepare the Environment
```bash
# Create & Activate Virtual Env
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/Mac

# Install Dependencies
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Anatomy

*   `models/` 🧠: Contains `stitchnet.py` (Architecture) and pre-trained `.pth` weights.
*   `stitch_auto.py` 🔢: The "brain" of the classical optimization pipeline.
*   `train_stitch_v2.py` 🎓: Advanced training script for fine-tuning on new datasets.
*   `data/` 📁: Home for your input tiles (supports PANDA format).
*   `stitched_results/` 🖼️: Output directory for your final high-res panoramas.

---

## 📝 Release Notes

### 🚀 v1.2.0 (Latest Release)
*   **Beautified UX:** Total overhaul of the README and UI with intuitive emojis and diagrams.
*   **Auto-Detection+:** Improved grid inference logic to handle missing tiles in a sequence.
*   **Stability:** Fixed a bug in the feathering mask that caused faint lines at tile borders.

### 📈 v1.1.0
*   **AI Integration:** Introduced the ResNet18-Flow architecture for superior alignment.
*   **Web App:** Launched the Streamlit-based graphical interface.
*   **Batch Support:** Added ability to process entire ZIP archives of tiles.

### 🏁 v1.0.0
*   **Core Engine:** Initial implementation of classical cubic polynomial distortion removal.
*   **Optimization:** Gauss-Newton solver integration for overlap error minimization.

---

## 🤝 Contributing
Contributions make the world go round! 🌍 If you have a fix or a feature, feel free to fork and PR.

## 📄 License
Licensed under the MIT License - see [LICENSE](LICENSE) for details.

---
*Developed with ❤️ for the Digital Pathology & Microscopy Community.*