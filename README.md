# 🔬 Optical Image Aberration Removal & Stitching (StitchNet)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**StitchNet** is a deep learning-based tool designed to stitch high-resolution microscopy tiles into seamless panoramas. It addresses common optical aberrations and alignment artifacts found in whole-slide imaging (WSI), specifically tailored for datasets like PANDA.

The project offers two modes of operation:
1.  **AI-Powered Stitching (StitchNet):** Uses a ResNet18-based flow network to predict pixel-wise warps for perfect alignment.
2.  **Classical Optimization:** Uses cubic polynomial distortion fields and Gauss-Newton optimization for mathematically rigorous stitching without training data.

---

## 🏗️ System Architecture

```mermaid
graph TD;
    A[Input: Raw Tiles] --> B{Choose Method};
    B -->|AI Mode| C[StitchNet (ResNet18)];
    B -->|Classical Mode| D[Polynomial Distortion Opt];
    
    C --> E[Predict Flow Field];
    D --> F[Optimize Parameters];
    
    E --> G[Warp Tiles];
    F --> G;
    
    G --> H[Feathering & Blending];
    H --> I[Seamless Panorama];
    I --> J[Auto-Crop Borders];
    J --> K[Final Output];
```

**Textual Flow:**
```text
[Input Tiles Folder] 
       │
       ▼
   (Preprocessing)
   Identify Grid (Rows x Cols)
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
 [ Deep Learning Path ]                      [ Classical Path ]
       │                                             │
  Load StitchNet Model                       Define Distortion Model
 (ResNet18 Encoder)                         (Cubic Polynomials)
       │                                             │
  Predict Optical Flow                       Optimize Overlap Error
 (Ref Image vs Neighbor)                    (Gauss-Newton Solver)
       │                                             │
       └──────────────────────┬──────────────────────┘
                              ▼
                        Warp Images
                              │
                    Blend & Stitch Canvas
                              │
                   Auto-Crop Black Borders
                              │
                              ▼
                     [ Final Panorama ]
```

---

## 🚀 Features

*   **Deep Learning Based:** Robust to complex non-linear deformations using `StitchNet`.
*   **Interactive UI:** User-friendly **Streamlit** dashboard for drag-and-drop stitching.
*   **Auto-Grid Detection:** Automatically infers grid dimensions from filenames.
*   **Robust Handling:** Includes classical fallback (Scipy/OpenCV) for verification.
*   **Large Scale Support:** Efficiently handles large arrays of high-res tiles.

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/muhammadmahadazher/optical-image-aberration-Removel_using_StitchNet.git
    cd optical-image-aberration-Removel_using_StitchNet
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

### 1. Web Application (Streamlit)
The easiest way to use the tool.

```bash
streamlit run streamlit_app.py
```
*   **Upload:** Drag & drop your tile images or a `.zip` file containing them.
*   **Configure:** Upload custom weights or use the default `models/stitchnet_v2_ep10.pth`.
*   **Stitch:** Click "Stitch Tiles" and download the high-res result.

### 2. Training StitchNet
To train the model on your own dataset (e.g., PANDA tiles):

```bash
python train_stitch_v2.py --tiles_dir data/panda_tiles --epochs 10 --batch_size 8
```

### 3. Classical Inference (CLI)
To run the classical stitching algorithm on a directory:

```bash
python stitch_auto.py --input_dir data/sample_slide
```

---

## 📂 Project Structure

```text
.
├── models/
│   ├── stitchnet.py       # PyTorch Model Architecture (ResNet-Flow)
│   ├── losses.py          # Training loss functions
│   └── stitchnet_v2_ep10.pth # Pre-trained model weights
├── data/                  # Place your input datasets here
├── stitched_results/      # Output directory for results
├── main.py                # Main entry point for CLI experiments
├── stitch_auto.py         # Classical stitching implementation
├── streamlit_app.py       # Web Interface source code
├── train_stitch_v2.py     # Training script
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## 📝 Release Notes

### v1.0.0 (Initial Release)
*   ✅ **Core:** Implemented `StitchNet` architecture for optical aberration removal.
*   ✅ **UI:** Added Streamlit web interface for easy usage.
*   ✅ **Classic:** Integrated `scipy.optimize` pipeline for baseline comparisons.
*   ✅ **Auto-Crop:** Added smart cropping to remove black artifacts from stitched images.
*   ✅ **Docs:** Comprehensive documentation and setup guide.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bugs or feature enhancements.

## 📄 License
This project is licensed under the MIT License.