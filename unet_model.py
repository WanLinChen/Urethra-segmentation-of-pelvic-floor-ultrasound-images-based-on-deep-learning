#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet_model.py
-------------
Defines a U-Net architecture for binary image segmentation.

U-Net (Ronneberger et al., 2015) consists of:
  - Encoder (contracting path): extracts features via Conv2D + MaxPooling
  - Bottleneck: deepest feature representation
  - Decoder (expansive path): restores spatial resolution via transposed
    convolutions, with skip connections from the encoder for detail recovery
  - Output: single-channel sigmoid map (pixel-wise probability)
"""

import tensorflow as tf


def build_unet(img_height: int, img_width: int, img_channels: int = 3) -> tf.keras.Model:
    """
    Build and compile a U-Net model.

    Parameters
    ----------
    img_height : int
        Height of the input images in pixels.
    img_width : int
        Width of the input images in pixels.
    img_channels : int
        Number of input channels (default: 3 for RGB).

    Returns
    -------
    tf.keras.Model
        Compiled U-Net model ready for training.
    """

    # ------------------------------------------------------------------ #
    # Input & normalization                                                #
    # ------------------------------------------------------------------ #
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

    # Normalize pixel values from [0, 255] to [0, 1]
    s = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    # ------------------------------------------------------------------ #
    # Encoder (Contracting Path)                                          #
    # Each block: Conv2D → Dropout → Conv2D → MaxPooling                  #
    # Filter counts double at each level; spatial size halves.            #
    # ------------------------------------------------------------------ #

    # Block 1 — 16 filters
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # Block 2 — 32 filters
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Block 3 — 64 filters
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Block 4 — 128 filters
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # ------------------------------------------------------------------ #
    # Bottleneck — 256 filters (no pooling)                               #
    # ------------------------------------------------------------------ #
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c5)

    # ------------------------------------------------------------------ #
    # Decoder (Expansive Path)                                            #
    # Each block: TransposedConv (upsample) → concat skip → Conv2D × 2   #
    # ------------------------------------------------------------------ #

    # Block 6 — upsample to match c4
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])          # skip connection from block 4
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c6)

    # Block 7 — upsample to match c3
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])          # skip connection from block 3
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c7)

    # Block 8 — upsample to match c2
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])          # skip connection from block 2
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c8)

    # Block 9 — upsample to match c1 (original resolution)
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)  # skip connection from block 1
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c9)

    # ------------------------------------------------------------------ #
    # Output layer                                                        #
    # 1×1 conv with sigmoid → pixel-wise probability in [0, 1]           #
    # ------------------------------------------------------------------ #
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Build model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    # Compile: Adam optimizer, binary cross-entropy loss for binary segmentation
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Quick sanity check — print model summary
    model = build_unet(img_height=640, img_width=800)
    model.summary()