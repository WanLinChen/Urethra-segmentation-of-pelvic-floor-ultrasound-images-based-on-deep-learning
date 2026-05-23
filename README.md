# Urethra Segmentation of Pelvic Floor Ultrasound Images Based on Deep Learning

> Academic exchange project — National Tsing Hua University × Peking University, Summer 2022  
> Advisor: Prof. Luo Jiajia, Department of Biomedical Engineering, Peking University


## Background

**Pelvic Floor Dysfunction (PFD)** — encompassing pelvic organ prolapse and stress urinary incontinence — affects approximately 1 in 10 women and carries enormous medical costs (USD ~$1 billion/year in the US alone). Clinical diagnosis often relies on observing urethral movement under ultrasound during resting (Rest) and Valsalva maneuvers, but precise automated tools for this are lacking.

This project builds an **automated urethra segmentation pipeline** for 2D pelvic floor ultrasound images using a **U-Net** deep learning architecture, with the goal of helping clinicians more accurately assess anterior pelvic organ prolapse.


## Project Structure

```
├── data/
│   ├── pelvic_train/          # Training data (8 patient cases, 235 frames)
│   │   └── <case_id>/
│   │       ├── images/        # Raw ultrasound frames (.png)
│   │       └── masks/         # Binary segmentation masks (.png)
│   └── pelvic_test/           # Test data (2 patient cases, 30 frames)
│       └── <case_id>/
│           └── images/
├── unet_model.py              # U-Net model definition
├── train.py                   # Data loading, preprocessing, and training
├── predict.py                 # Run inference and visualize results
└── README.md
```


## Model Architecture — U-Net

U-Net is a fully convolutional network originally proposed for biomedical image segmentation. It consists of two symmetric paths:

- **Encoder (Contracting Path):** Successive Conv2D + ReLU + MaxPooling blocks progressively extract features while reducing spatial dimensions. Filter counts: 16 → 32 → 64 → 128 → 256.
- **Bottleneck:** Deepest feature representation (256 channels).
- **Decoder (Expansive Path):** Transposed convolutions upsample the feature maps back to the original resolution. Skip connections from the encoder are concatenated at each level to preserve spatial detail.
- **Output:** A single-channel sigmoid activation map — each pixel receives a probability of belonging to the urethra.

<img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/7c87da11-f966-45d1-a3ff-8ac4534ffae4" />


**Training config:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metric: Accuracy
- Input size: 800 × 640 × 3
- Dropout: 0.1–0.3 (regularization)


## Dataset

| Split      | Cases | Frames |
|------------|-------|--------|
| Training   | 6     | 176    |
| Validation | 2     | 59     |
| Test       | 2     | 30     |
| **Total**  | **10**| **265**|

- Source: Peking University People's Hospital
- Modality: 2D B-mode transverse pelvic ultrasound (Mindray DC-3X / DB-2U)
- Annotation: Manual urethra labeling performed with **3D Slicer**, exported as binary label maps (masks)
- Images converted from `.dcm` → `.png`; resized from 1620×910 / 1920×910 → **800×640**


## Setup

```bash
# Clone the repository
git clone https://github.com/WanLinChen/Urethra-segmentation-of-pelvic-floor-ultrasound-images-based-on-deep-learning.git
cd Urethra-segmentation-of-pelvic-floor-ultrasound-images-based-on-deep-learning

# Install dependencies
pip install tensorflow numpy scikit-image matplotlib tqdm Pillow opencv-python
```

**Tested with:** Python 3.8, TensorFlow 2.x


## Usage

### 1. Prepare Data
Organize your data under `data/pelvic_train/` and `data/pelvic_test/` following the structure above. Each case folder should contain an `images/` subfolder and (for training) a `masks/` subfolder.

### 2. Train
```bash
python train.py
```
This loads all training images and masks, resizes them to 800×640, trains the U-Net for the configured number of epochs, and saves the best model weights.

### 3. Predict & Visualize
```bash
python predict.py
```
Runs inference on the test set and generates side-by-side comparison plots (original image | predicted mask | ground truth mask).


## Results

The model successfully locates the urethra in most images across all three splits. Qualitative examples below show the original ultrasound frame (with the urethra circled in red), the model's predicted mask, and the manually annotated ground truth mask.

| Set        | Observation |
|------------|-------------|
| Training   | Strong localization; shape fidelity improving with training data exposure |
| Validation | Good position detection; occasional shape/size mismatch |
| Test       | Correct region identified; some over-segmentation of adjacent structures |

**Known limitations:**
- Small dataset (265 frames total) limits generalization
- Some false positive regions segmented in test images (low-contrast areas)
- Shape accuracy on test set lags behind training/validation sets

**Potential improvements:**
- Data augmentation (flipping, rotation, elastic deformation)
- Larger and more diverse dataset
- Experimenting with UNet++ or ResNet-based encoders


## Dependencies

| Package | Purpose |
|---------|---------|
| TensorFlow 2.x | Model building & training |
| NumPy | Array operations |
| scikit-image | Image I/O and resizing |
| Pillow | Mask loading |
| OpenCV | Image preprocessing |
| Matplotlib | Result visualization |
| tqdm | Progress bars |


## Reference

- Ronneberger O., Fischer P., Brox T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv:1505.04597](https://arxiv.org/pdf/1505.04597.pdf)
- 3D Slicer: https://slicer.readthedocs.io/en/latest/user_guide/about.html


## Author

**陳宛琳 (Wan-Lin Chen)**  
Department of Engineering and System Science, National Tsing Hua University  
2022 Cross-Strait Summer Academic Exchange Program
## Acknowledgements

This project was conducted as part of a cross-strait summer academic exchange program between National Tsing Hua University and Peking University. Special thanks to Professor Jiajia Luo from the Department of Biomedical Engineering at Peking University for guidance and supervision.

