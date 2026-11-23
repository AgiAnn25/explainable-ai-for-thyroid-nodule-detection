# Explainable AI for Thyroid Nodule Detection Using Ultrasound Images

## Overview

This project focuses on binary classification of thyroid ultrasound images using a custom Convolutional Neural Network (CNN) architecture called **Fibonacci-Net**. The model is designed with Fibonacci-inspired filter configurations to extract image features at different levels of complexity.

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is integrated to visualize the regions of ultrasound images that contribute to the model's predictions, providing an interpretable view of the classification process.

## Objectives

- Perform exploratory data analysis on thyroid ultrasound images.
- Preprocess and prepare images for deep learning.
- Develop a custom Fibonacci-inspired CNN architecture.
- Train and evaluate the model for binary image classification.
- Visualize model predictions using Grad-CAM.
- Explore explainability in medical image classification.

## Dataset

The dataset consists of **3,115 original thyroid ultrasound images** organized into two classes.

| Class | Number of Images |
|---|---:|
| Class 0 | 1,905 |
| Class 1 | 1,210 |
| **Total** | **3,115** |

The dataset is organized into folders corresponding to the two classes.

```text
Thyroid Data/
├── 0/
└── 1/
```

## Model Architecture – Fibonacci-Net

**Fibonacci-Net** is a custom CNN architecture inspired by the Fibonacci sequence. It utilizes Fibonacci-inspired filter configurations of **21, 34, 55, 89, 144, 233, and 377** across its convolutional blocks to extract image features at different levels of complexity.

The architecture incorporates:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Parallel feature connections (PCB)
- Custom Avg2MaxPooling
- Depthwise Separable Convolution
- Global Average Pooling
- Sigmoid activation for binary classification

## Explainability – Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to generate heatmaps highlighting image regions associated with the model's predictions.

These heatmaps are overlaid on the original ultrasound images to visually explore the regions influencing the classification output and gain insight into the model's decision-making process.

## Workflow

1. Dataset Loading
2. Dataset Inspection
3. Exploratory Data Analysis (EDA)
4. Data Balancing
5. Train, Validation, and Test Split
6. Image Preprocessing
7. Fibonacci-Net Architecture Development
8. Model Training
9. Training Visualization
10. Model Evaluation
11. Model Saving
12. Grad-CAM Explainability

## Model Evaluation

The model was evaluated on a held-out test dataset containing **381 ultrasound images** using classification metrics to assess its performance across both classes.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.84 | 0.84 | 0.84 | 191 |
| Positive | 0.84 | 0.84 | 0.84 | 190 |
| **Accuracy** | | | **0.84** | **381** |
| Macro Average | 0.84 | 0.84 | 0.84 | 381 |
| Weighted Average | 0.84 | 0.84 | 0.84 | 381 |

### Performance Summary

| Metric | Score |
|---|---:|
| Accuracy | 84% |
| Precision | 84% |
| Recall | 84% |
| F1-Score | 84% |

The model achieved consistent precision, recall, and F1-scores across both classes, indicating balanced classification performance on the test dataset.

## Visualizations

The project includes visualizations to understand the dataset, training process, model performance, and explainability.

- Dataset distribution
- Training and validation accuracy
- Training and validation loss
- Confusion matrix
- ROC curve
- Grad-CAM heatmaps

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn

## Project Structure

```text
Explainable-AI-for-Thyroid-Nodule-Detection/
│
├── Thyroid Data/
│   ├── 0/
│   └── 1/
│
├── notebooks/
│   └── Explainable_AI_Thyroid_Nodule_Detection.ipynb
│
├── models/
│   └── Thyroid_cancer_model.keras
│
├── README.md
└── requirements.txt
```

## Model Saving

The trained Fibonacci-Net model is saved in the Keras `.keras` format, preserving the model architecture and learned weights, along with its training configuration and optimizer state when available.

```python
model.save("../models/Thyroid_cancer_model.keras")
```

## Conclusion

This project explored thyroid ultrasound image classification using a custom CNN architecture, **Fibonacci-Net**, incorporating Fibonacci-inspired filter configurations. The model achieved **84% accuracy** on the test dataset, with balanced precision, recall, and F1-score across both classes.

The integration of **Grad-CAM** provided visual insights into image regions contributing to the model's predictions, adding an explainability component to the classification pipeline. This project demonstrates the application of deep learning and explainable AI techniques in medical image analysis while highlighting the importance of interpretable model predictions.

Further improvements could explore architectural optimization, larger and more diverse datasets, and additional validation to enhance model performance and reliability.

