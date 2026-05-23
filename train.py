#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
--------
Data loading, preprocessing, and U-Net training for urethra segmentation
in pelvic floor ultrasound images.

Dataset layout expected:
    pelvic_train/
        <case_id>/
            images/   <frame_number>.png
            masks/    <frame_number>-modified.png
    pelvic_test/
        <case_id>/
            images/   <frame_number>.png

Usage:
    python train.py
"""

import os
import random

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from unet_model import build_unet


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

IMG_HEIGHT   = 640    # Target height after resize
IMG_WIDTH    = 800    # Target width after resize
IMG_CHANNELS = 3      # RGB input

TRAIN_PATH = 'data/pelvic_train/'
TEST_PATH  = 'data/pelvic_test/'

EPOCHS     = 50
BATCH_SIZE = 8
VAL_SPLIT  = 0.25    # Fraction of training data used for validation

MODEL_SAVE_PATH = 'unet_urethra.h5'


# ------------------------------------------------------------------ #
# Data Loading                                                         #
# ------------------------------------------------------------------ #

def load_training_data(train_path: str) -> tuple:
    """
    Read and resize all training images and their corresponding binary masks.

    Parameters
    ----------
    train_path : str
        Path to the training data root directory.

    Returns
    -------
    X_train : np.ndarray, shape (N, H, W, 3), dtype uint8
        Resized RGB ultrasound frames.
    Y_train : np.ndarray, shape (N, H, W, 1), dtype bool
        Corresponding binary segmentation masks.
    """
    train_ids = next(os.walk(train_path))[1]  # One subfolder per patient case

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    print(f'Loading {len(train_ids)} training samples ...')

    for n, case_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        case_path = os.path.join(train_path, case_id)

        # --- Image ---
        # Frame number is encoded in the folder name (characters 11–13)
        frame_num = int(case_id[11:14]) + 1
        img_path  = os.path.join(case_path, 'images', f'{frame_num}.png')
        img = imread(img_path)[:, :, :IMG_CHANNELS]  # Drop alpha channel if present
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True).astype(np.uint8)
        X_train[n] = img

        # --- Mask ---
        mask_num  = int(case_id[11:14])
        mask_path = os.path.join(case_path, 'masks', f'{mask_num}-modified.png')
        mask = Image.open(mask_path).convert('L')        # Load as grayscale
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)             # Add channel dim → (H, W, 1)
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH),
                      mode='constant', preserve_range=True)
        Y_train[n] = mask > 0                            # Binarize

    return X_train, Y_train


def load_test_data(test_path: str) -> tuple:
    """
    Read and resize all test images (no masks available).

    Parameters
    ----------
    test_path : str
        Path to the test data root directory.

    Returns
    -------
    X_test : np.ndarray, shape (N, H, W, 3), dtype uint8
    original_sizes : list of (height, width) tuples
        Original image dimensions before resizing (useful for up-scaling predictions).
    """
    test_ids = next(os.walk(test_path))[1]

    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    original_sizes = []

    print(f'Loading {len(test_ids)} test samples ...')

    for n, case_id in tqdm(enumerate(test_ids), total=len(test_ids)):
        case_path = os.path.join(test_path, case_id)
        frame_num = int(case_id[11:14])
        img_path  = os.path.join(case_path, 'images', f'{frame_num}.png')

        img = imread(img_path)[:, :, :IMG_CHANNELS]
        original_sizes.append((img.shape[0], img.shape[1]))

        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True).astype(np.uint8)
        X_test[n] = img

    print('Done.')
    return X_test, original_sizes


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #

def train():
    # Load data
    X_train, Y_train = load_training_data(TRAIN_PATH)

    # Build U-Net
    model = build_unet(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                       img_channels=IMG_CHANNELS)
    model.summary()

    # Callbacks
    callbacks = [
        # Save the model whenever validation loss improves
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Stop early if val_loss does not improve for 10 consecutive epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # Train
    history = model.fit(
        X_train, Y_train,
        validation_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Plot training curves
    plot_training_history(history)

    return model, history


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Plot and save the loss and accuracy curves from training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Cross-Entropy')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print('Training curves saved to training_curves.png')


if __name__ == '__main__':
    model, history = train()
    print(f'Best model saved to {MODEL_SAVE_PATH}')