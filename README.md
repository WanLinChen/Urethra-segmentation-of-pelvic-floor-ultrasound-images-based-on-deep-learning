# Urethra Segmentation in Pelvic Floor Ultrasound Images

## Project Overview

This project aims to develop an automated system for segmenting the urethra in female pelvic floor ultrasound images using deep learning techniques. The goal is to assist gynecologists in diagnosing pelvic organ prolapse conditions more accurately and efficiently.

## Motivation

Pelvic floor dysfunction (PFD) affects approximately 1 in 10 women, significantly impacting their quality of life. Accurate diagnosis often requires analyzing ultrasound images of the pelvic region. By automating the segmentation of the urethra, this project seeks to:

1. Reduce the manual workload for medical professionals
2. Improve the precision of pelvic organ prolapse assessments
3. Potentially lower treatment costs associated with PFD

## Methodology

The project utilizes the following approach:

1. Data Collection: 10 sets of female pelvic ultrasound images from Peking University People's Hospital
2. Data Annotation: Manual labeling of the urethra in each image frame using 3D Slicer software
3. Model Architecture: Implementation of a U-Net deep learning network for image segmentation
4. Training and Validation: Using 8 image sets for training (235 images total, with 59 for validation)
5. Testing: Evaluation of the model using 2 separate image sets (30 images)

## Results

The current model demonstrates the ability to roughly locate the urethra in most images. However, there are limitations in terms of shape and size accuracy compared to manual annotations. The project identifies areas for potential improvement, including:

1. Increasing the training dataset size
2. Exploring alternative network architectures
3. Enhancing the model's generalization capabilities

## Acknowledgements

This project was conducted as part of a cross-strait summer academic exchange program between National Tsing Hua University and Peking University. Special thanks to Professor Jiajia Luo from the Department of Biomedical Engineering at Peking University for guidance and supervision.

