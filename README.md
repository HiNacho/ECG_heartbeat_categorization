# ECG_heartbeat_categorization
This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing some of the capabilities of transfer learning on it. 
# 🔬 Advanced ECG Heartbeat Classification Model Summary

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Status](https://img.shields.io/badge/Status-Model%20Trained-success)
![F1 Score](https://img.shields.io/badge/Validation%20Macro%20F1-98.07%25-brightgreen)

---

## 🎯 Project Overview

**Project:** *Single-Modal 1D CNN–LSTM Hybrid for Arrhythmia Classification*  
**Goal:** Achieve high diagnostic performance (**F1-Score**) across all five heartbeat classes —  
**Normal (N)** · **Supraventricular (S)** · **Ventricular (V)** · **Fusion (F)** · **Unknown (Q)**  

> 🩺 Focus: Improve **Precision** in minority classes (S and F) while maintaining strong Recall and overall balance across all categories.

---

## 🧠 1. Model Architecture & Methodology

| **Category** | **Component** | **Rationale for Improvement** |
|---------------|---------------|-------------------------------|
| 🏗️ **Architecture** | **1D CNN–LSTM Hybrid** | Combines CNN layers for extracting beat morphology (shape) features with Bidirectional LSTM layers for capturing temporal context (rhythm and RR-interval patterns), providing a more holistic classification. |
| 📊 **Data Strategy** | **SMOTE Oversampling** | Partially balances the training data by synthesizing examples of minority classes (S, V, F, Q). This reduces the model's bias toward the majority class (Normal) and improves the quality of learned features for rare beats. |
| ⚖️ **Loss Function** | **Custom Weighted CrossEntropyLoss** | Uses manually tuned weights based on the SMOTE-oversampled distribution. This applies targeted penalties against False Negatives (high Recall focus) without overly damaging Precision. |
| ⚙️ **Optimization** | **Adam Optimizer + ReduceLROnPlateau** | Standard deep learning optimizer combined with a learning rate scheduler to prevent the model from getting stuck in local minima and ensure stable convergence. |

---

## 📈 2. Training Performance Snapshot

> The model was trained for **30 epochs** with **early stopping (patience = 12)**, monitoring the **Validation Macro F1-Score**.

| **Metric** | **Last Epoch (30) Result** | **Trend / Insight** |
|-------------|----------------------------|----------------------|
| 🧩 **Train Loss** | `0.0028` | Strong convergence on training data — low residual error. |
| 📉 **Validation Loss** | `0.9341` | Large gap between train/validation indicates overfitting. |
| 🏅 **Validation Macro F1** | `0.9807 (98.07%)` | Excellent generalization — strong and balanced class performance. |
| ⏹️ **Convergence** | No improvement for 4 epochs | Training stopped at the performance plateau (optimal stopping). |

---

## 🧪 3. Final Test Evaluation

**Action Required:**  
Run the following function to obtain the **final unbiased test performance** metrics:

```python
evaluate_model(model, test_loader)

