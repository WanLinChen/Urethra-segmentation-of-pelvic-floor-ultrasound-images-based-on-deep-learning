#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
----------
Load a trained U-Net model and run inference on test images.
Generates side-by-side visualizations:
    Original image | Predicted mask | Ground-truth mask (if available)

Usage:
    python predict.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

from train import load_test_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from train import TRAIN_PATH, TEST_PATH


# ------------------------------------------------------------------
# Configuration                                                       
# ------------------------------------------------------------------ 

MODEL_PATH       = 'unet_urethra.h5'   # Trained model weights
THRESHOLD        = 0.5                  # Sigmoid threshold for binary prediction
OUTPUT_DIR       = 'predictions/'       # Folder to save visualizations
SAMPLES_TO_SHOW  = 5                    # Number of random samples to visualize


# ------------------------------------------------------------------ 
# Helpers                                                              
# ------------------------------------------------------------------ 

def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    print(f'Loading model from {model_path} ...')
    model = tf.keras.models.load_model(model_path)
    print('Model loaded.')
    return model


def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Run model inference and threshold the sigmoid output.

    Parameters
    ----------
    model : tf.keras.Model
    X : np.ndarray, shape (N, H, W, 3)

    Returns
    -------
    np.ndarray, shape (N, H, W, 1), dtype bool
        Binary predicted masks.
    """
    preds_raw = model.predict(X, verbose=1)   # Values in [0, 1]
    preds_bin = preds_raw > THRESHOLD          # Binarize
    return preds_bin


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray,
                           color: tuple = (255, 200, 0), alpha: float = 0.45) -> np.ndarray:
    """
    Blend a binary mask onto an RGB image with a semi-transparent color overlay.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), uint8
    mask  : np.ndarray (H, W) or (H, W, 1), bool
    color : RGB tuple for the mask overlay (default: yellow)
    alpha : overlay transparency (0 = transparent, 1 = opaque)

    Returns
    -------
    np.ndarray (H, W, 3), uint8
    """
    mask = mask.squeeze().astype(bool)
    overlay = image.copy().astype(np.float32)
    for c, val in enumerate(color):
        overlay[mask, c] = overlay[mask, c] * (1 - alpha) + val * alpha
    return overlay.astype(np.uint8)


def visualize_results(X: np.ndarray,
                       preds: np.ndarray,
                       Y_true: np.ndarray = None,
                       indices: list = None,
                       title_prefix: str = 'Sample',
                       save_dir: str = OUTPUT_DIR) -> None:
    """
    Plot and save side-by-side comparisons for selected samples.

    Parameters
    ----------
    X           : Input images, (N, H, W, 3)
    preds       : Predicted masks, (N, H, W, 1)
    Y_true      : Ground-truth masks (optional), (N, H, W, 1)
    indices     : List of sample indices to visualize (None → random selection)
    title_prefix: Label prefix for figure titles
    save_dir    : Directory to write PNG files
    """
    os.makedirs(save_dir, exist_ok=True)

    if indices is None:
        indices = np.random.choice(len(X), size=min(SAMPLES_TO_SHOW, len(X)), replace=False)

    n_cols = 3 if Y_true is not None else 2

    for idx in indices:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        fig.suptitle(f'{title_prefix} #{idx}', fontsize=14)

        # --- Original image ---
        axes[0].imshow(X[idx])
        axes[0].set_title('Original Ultrasound')
        axes[0].axis('off')

        # --- Predicted mask overlay ---
        pred_overlay = overlay_mask_on_image(X[idx], preds[idx], color=(255, 200, 0))
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Predicted Mask (yellow)')
        axes[1].axis('off')

        # --- Ground-truth mask overlay (if provided) ---
        if Y_true is not None:
            gt_overlay = overlay_mask_on_image(X[idx], Y_true[idx], color=(255, 200, 0))
            axes[2].imshow(gt_overlay)
            axes[2].set_title('Ground-Truth Mask (yellow)')
            axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{title_prefix.lower().replace(" ", "_")}_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_path}')


# ------------------------------------------------------------------ 
# Dice Coefficient (optional evaluation metric)                        
# ------------------------------------------------------------------ 

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                      smooth: float = 1e-6) -> float:
    """
    Compute the Dice similarity coefficient between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    A value of 1.0 means perfect overlap; 0.0 means no overlap.

    Parameters
    ----------
    y_true : np.ndarray, bool
    y_pred : np.ndarray, bool
    smooth : small constant to avoid division by zero

    Returns
    -------
    float
    """
    y_true_f = y_true.flatten().astype(float)
    y_pred_f = y_pred.flatten().astype(float)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# ------------------------------------------------------------------ 
# Main                                                                 
# ------------------------------------------------------------------ 

def main():
    # Load model
    model = load_model(MODEL_PATH)

    # ----- Test set -----
    X_test, _ = load_test_data(TEST_PATH)
    preds_test = predict(model, X_test)
    visualize_results(X_test, preds_test,
                      title_prefix='Test', save_dir=os.path.join(OUTPUT_DIR, 'test'))

    # ----- Training / validation set (with ground-truth masks) -----
    # Reload training data to compare predictions against labels
    from train import load_training_data
    X_train, Y_train = load_training_data(TRAIN_PATH)
    preds_train = predict(model, X_train)

    # Compute mean Dice score over training samples
    dice_scores = [
        dice_coefficient(Y_train[i], preds_train[i])
        for i in range(len(X_train))
    ]
    print(f'\nMean Dice (training set): {np.mean(dice_scores):.4f} '
          f'± {np.std(dice_scores):.4f}')

    visualize_results(X_train, preds_train, Y_true=Y_train,
                      title_prefix='Train', save_dir=os.path.join(OUTPUT_DIR, 'train'))

    print(f'\nAll visualizations saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()