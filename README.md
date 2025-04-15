# SSDGL - Spectral-Spatial-Dependent Global Learning Framework for Hyperspectral Image Classification

This project implements the full pipeline described in the paper:  
"A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification" using TensorFlow and Indian Pines dataset.

The method combines:
- Spectral-Spatial Dual Branch CNNs
- Adaptive Attention-based Fusion
- Global Context Learning (GCL)
- Class-Balanced Training for Imbalanced Labels

---

## Features

- Implemented in Python 3 + TensorFlow 2.x
- Supports Indian Pines dataset (145×145×200)
- Handles class imbalance using scikit-learn’s class_weight
- Outputs accuracy, confusion matrix, F1, precision, recall & Kappa
- Includes classification visualization over hyperspectral images

---

## Dataset: Indian Pines

We use the publicly available Indian Pines dataset:

- Data: `Indian_pines_corrected.mat`  
- Ground Truth: `Indian_pines_gt.mat`

Automatically downloaded in the notebook if not found locally.

---

## Installation

Make sure you’re using Google Colab or a local Python environment with TensorFlow. Then:

```bash
pip install tensorflow scikit-learn matplotlib spectral
```

---

## Running the Project

Use the notebook or Python script and run all cells in order:

1. Load and normalize HSI data
2. Extract 3D patches (11×11)
3. Build SSDGL model with dual-branch CNN + attention
4. Train with class-balanced loss
5. Predict and visualize results

---

## Model Architecture

- Spectral Branch → 1D CNN over spectral dimension
- Spatial Branch → 2D CNN over spatial dimension
- Adaptive Fusion → Dense + softmax attention weights
- Global Context → Dense + softmax-based enhancement
- Classifier → Fully connected layers with softmax

---

## Evaluation Metrics

- Overall Accuracy (OA)
- Cohen’s Kappa Score
- Macro F1 Score
- Macro Precision & Recall
- Confusion Matrix (visual)
- Classification Map (2D RGB overlay)

---

## Visualizations

- Ground Truth vs Predicted Class Maps
- Confusion Matrix heatmap
- Accuracy/loss training curves

