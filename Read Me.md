# 🌲 Forest CoverType Classification  
## Deep Learning Optimization & Ensemble Benchmarking

---

## 📌 Executive Summary

This project evaluates the performance of a Multi-Layer Perceptron (MLP) against a Random Forest classifier on the UCI Forest CoverType dataset, a large-scale structured multi-class classification problem.

The goal was not only to improve neural network performance but to critically assess whether deep learning provides an advantage over tree-based ensemble methods for tabular data.

This project was completed as part of the Deep Learning module at AtomCamp.

---

## 📊 Dataset Overview

- **Dataset:** UCI Forest CoverType  
- **Source:** Available via `sklearn.datasets.fetch_covtype`  
- **Instances:** 581,012  
- **Features:** 54 cartographic attributes  
- **Target Classes:** 7 forest cover types  
- **Problem Type:** Multi-class classification  
- **Challenge:** Class imbalance across categories  

The dataset includes cartographic variables such as elevation, slope, soil type, and wilderness area.

---

## 🎯 Project Objectives

- Improve a baseline Multi-Layer Perceptron (MLP)
- Apply architecture tuning and regularization techniques
- Experiment with optimizers and learning rate strategies
- Evaluate using multiple metrics beyond accuracy
- Compare performance against a Random Forest classifier
- Analyze why ensemble models often outperform neural networks on structured data

---

# 🧠 Model 1 — Optimized Multi-Layer Perceptron (MLP)

## Architecture Enhancements

- Deeper and wider dense layers
- ReLU and LeakyReLU activation functions
- Batch Normalization layers
- Dropout for regularization
- L2 weight decay

## Optimization Strategy

- Optimizers tested:
  - Adam
  - RMSprop
  - SGD with Momentum
- Learning Rate Scheduling:
  - ReduceLROnPlateau
- EarlyStopping to prevent overfitting

Training and validation curves were monitored to diagnose underfitting and overfitting.

---

# 🌲 Model 2 — Random Forest Classifier

- Ensemble-based decision tree method
- Captures non-linear relationships effectively
- Less sensitive to feature scaling
- Strong performance on structured/tabular datasets
- Used as a benchmarking baseline

---

# 📈 Evaluation Metrics

Due to class imbalance, model performance was evaluated using:

- Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrix
- Training & Validation Learning Curves

This ensured balanced performance analysis across all classes.

---

# 🔍 Comparative Insights

| Aspect | MLP | Random Forest |
|--------|------|---------------|
| Tuning Complexity | High | Moderate |
| Sensitivity to Scaling | High | Low |
| Feature Interaction Handling | Learned | Inherent |
| Generalization on Tabular Data | Competitive | Strong |

### Key Insight

Tree-based ensemble methods often outperform neural networks on structured/tabular datasets because they naturally model feature interactions and non-linear boundaries without extensive hyperparameter tuning.

Neural networks require careful optimization and regularization to achieve competitive performance.

---

# 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

# 🚀 How to Run

1. Clone the repository:

```bash
git clone <your-repository-link>
cd <repository-name>