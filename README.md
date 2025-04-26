# CAPTCHA Recognition System

## Overview
This project focuses on developing a CAPTCHA recognition system using machine learning. It involves tokenizing CAPTCHA images to segment them into individual characters, training a Convolutional Neural Network (CNN) to recognize characters, and predicting the full CAPTCHA text from new images.

## Features
- **Segmentation**: Developed a robust pipeline for character segmentation using adaptive thresholding, contour detection, and optimized region-based and color-based methods.
- **Recognition Model**: Built a CNN capable of accurately recognizing alphanumeric characters (0–9, a–z).
- **Data Cleaning**: Cleaned mislabeled and problematic images to improve model performance.
- **Augmentation**: Employed an augmentation pipeline to increase dataset diversity and reduce overfitting.

## Approach

### Baseline Segmentation
- Adaptive Thresholding & Inversion
- Median Blurring and Morphological Closing
- Contour Detection & Filtering
- Width Analysis to manage overlapping characters

### Preprocessing
- Grayscale conversion and contrast enhancement
- Gaussian adaptive thresholding for binarization
- Median filtering and morphological opening to clean noise
- Contour detection and bounding box calculation for text regions

### Optimized Segmentation
- **Region-based segmentation**: Grouping overlapping character regions using a 70% overlap threshold.
- **Color-based segmentation**: Analyzing hue transitions in the HSV color space to identify characters.

### Model
- CNN with standardized input dimensions
- One-hot encoded labels for character classification
- Augmentation pipeline to introduce synthetic variance

### Data Cleaning
- Validated segmented character counts against labels
- Used Tesseract OCR as fallback validation
- Manually inspected and corrected ~550 problematic training images

## Results
- **Training loss**: ~0.1552
- **Validation loss**: ~0.8350
- **Model Precision, Recall, F1 Score**: ~86%
- Character-wise accuracy measured using a confusion matrix

## Limitations
- Sensitive to extreme image skewness
- Challenges in segmenting multiple connected characters
- High variability in font styles, colors, and noise patterns

## Contributions
- **Pawan Kishor Patil**: Code structure, noise removal research, model training and optimization, dataset cleaning
- **Cyrus Krispin Vijikumar**: Segmentation development, model training and testing
- **Suresh Vaidyanaath**: Data preprocessing, segmentation pipeline design and implementation
- **Arkobrata Chaudhuri**: Dataset cleaning (automation and manual), segmentation refinement

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Tesseract OCR
