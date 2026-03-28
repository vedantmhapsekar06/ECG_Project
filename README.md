# ECG_Project
# 🫀 ECG Signal Classification using Machine Learning

> **Internship Project** — Electronics & Telecommunication Engineering
> **Mentor:** Ameya K. Naik and Nitin S. Nagori
> **Goal:** Develop an ML model to classify ECG signals into 5 cardiac conditions, deployable on a low-power portable device (Arduino-based)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Output Files](#-output-files)
- [ML Models Used](#-ml-models-used)
- [Results](#-results)
- [Technologies Used](#-technologies-used)

---

## 📌 Project Overview

This project builds a Machine Learning pipeline to **analyze and classify ECG signals** based on cardiac abnormalities. The model is designed to eventually run on a **low-power Arduino-based portable device**.

### ECG Conditions Detected:
| Label | Condition |
|-------|-----------|
| 0 | Arrhythmia |
| 1 | Heart Failure |
| 2 | Myocardial Infarction |
| 3 | Normal |
| 4 | Stress |

---

## 📁 Dataset

- **File:** `imbalanced_ecg_dataset.csv`
- **Rows:** 500 patients
- **Features:** 16 ECG features (Age, Heart Rate, RR Interval, QRS Duration, ST Elevation, HRV, Blood Pressure, etc.)
- **Target:** `Condition` (5 classes)
- **Challenge:** Imbalanced classes (Normal: 250, Arrhythmia: 100, Stress: 80, Heart Failure: 40, Myocardial Infarction: 30)
- **Solution:** SMOTE (Synthetic Minority Oversampling Technique)

---

## 📂 Project Structure
```
ECG_Project/
│
├── imbalanced_ecg_dataset.csv          # 1. Raw dataset
│
├── load_preprocess_features.py         # 2. Data loading, preprocessing & feature engineering
├── ml_model_randomforest.py            # 3. ML Model 1 — Random Forest
├── ml_model_svm.py                     # 4. ML Model 2 — Support Vector Machine (SVM)
├── model_compare.py                    # 5. Compare both models & select best
├── model_evaluation.py                 # 6. Deep evaluation of the best model
│
└── outputs/                            # All saved models, plots & results
    ├── X_scaled.npy
    ├── y_resampled.npy
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_names.pkl
    ├── rf_model.pkl
    ├── svm_model.pkl
    ├── rf_scores.csv
    ├── svm_scores.csv
    ├── best_model.csv
    ├── final_summary.csv
    └── [plots — see Output Files section below]
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vedantmhapsekar06/ECG_Project.git
cd ECG_Project
```

### 2. Install Python
Download from [python.org](https://www.python.org/downloads/) — install with **"Add to PATH"** ✅ checked.

### 3. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

---

## ▶️ How to Run

> ⚠️ **Run files in order** — each file saves outputs used by the next one.

Open terminal in VS Code (`Ctrl + ~`) inside the `ECG_Project` folder:

### Step 1 — Load, Preprocess & Feature Engineering
```bash
python load_preprocess_features.py
```
Loads dataset → encodes features → applies SMOTE → scales features → saves processed data.

### Step 2 — Train Random Forest
```bash
python ml_model_randomforest.py
```
Trains Random Forest → evaluates → saves model and scores.

### Step 3 — Train SVM
```bash
python ml_model_svm.py
```
Trains SVM → evaluates → saves model and scores.

### Step 4 — Compare Both Models
```bash
python model_compare.py
```
Compares RF vs SVM on all metrics → saves best model name to `outputs/best_model.csv`.

### Step 5 — Final Model Evaluation
```bash
python model_evaluation.py
```
Loads the best model → generates ROC curves, learning curves, confusion matrix, F1 per class.

---

## 📊 Output Files

### `load_preprocess_features.py` generates:
| File | Description |
|------|-------------|
| `outputs/smote_balancing.png` | Class distribution Before vs After SMOTE |

### `ml_model_randomforest.py` generates:
| File | Description |
|------|-------------|
| `outputs/rf_confusion_matrix.png` | Confusion matrix for Random Forest |
| `outputs/rf_feature_importance.png` | Top ECG features by importance score |
| `outputs/rf_scores.csv` | Accuracy, F1, ROC-AUC, CV scores |

### `ml_model_svm.py` generates:
| File | Description |
|------|-------------|
| `outputs/svm_confusion_matrix.png` | Confusion matrix for SVM |
| `outputs/svm_scores.csv` | Accuracy, F1, ROC-AUC, CV scores |

### `model_compare.py` generates:
| File | Description |
|------|-------------|
| `outputs/comparison_bar_chart.png` | Side-by-side bar chart of all metrics |
| `outputs/comparison_radar_chart.png` | Radar chart of overall model performance |
| `outputs/best_model.csv` | Name of the winning model |

### `model_evaluation.py` generates:
| File | Description |
|------|-------------|
| `outputs/final_confusion_matrix.png` | Final confusion matrix of best model |
| `outputs/final_roc_curve.png` | ROC curves per class with AUC scores |
| `outputs/final_learning_curve.png` | Training vs validation accuracy curve |
| `outputs/final_f1_per_class.png` | F1 score per ECG condition |
| `outputs/final_summary.csv` | Final accuracy, F1, ROC-AUC summary |

---

## 🤖 ML Models Used

### 1. 🌲 Random Forest
- Ensemble of 200 decision trees
- Handles imbalanced data with `class_weight='balanced'`
- Provides feature importance scores
- Scale-invariant, fast training

### 2. ⚡ Support Vector Machine (SVM)
- RBF (Radial Basis Function) kernel
- Strong generalization on small, well-scaled datasets
- Evaluated with probability outputs for ROC-AUC

---

## 📈 Results

| Metric | Random Forest | SVM |
|--------|-------------|-----|
| Accuracy | 81.20% | **84.00%** |
| F1 Macro | 0.80 | **0.83** |
| F1 Weighted | 0.80 | **0.83** |
| ROC-AUC | 0.96 | **0.97** |
| CV Accuracy | 81.68% | **83.36%** |

> 🏆 **Best Model: SVM** — won across all 5 evaluation metrics

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML models, preprocessing, evaluation |
| Imbalanced-learn | SMOTE for class balancing |
| Matplotlib | Plotting graphs |
| Seaborn | Statistical visualizations |
| Joblib | Saving and loading models |

---

## 👨‍💻 Author

**Vedant Mhapsekar**
Internship under: Ameya K. Naik & Nitin S. Nagori

---

## 📄 License

This project is for academic/internship purposes.
