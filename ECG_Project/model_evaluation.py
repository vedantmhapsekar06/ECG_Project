
# FILE 6: Final Model Evaluation
# Uses the BEST MODEL chosen in File 5 (5_compare.py)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

os.makedirs('outputs', exist_ok=True)


# 1. LOAD BEST MODEL — AS DECIDED IN FILE 5

best_model_name = pd.read_csv('outputs/best_model.csv').iloc[0]['best_model']

print("=" * 55)
print("   FILE 6: FINAL MODEL EVALUATION")
print("=" * 55)
print(f"  ✅ Best Model from File 5 : {best_model_name}")

if best_model_name == 'Random Forest':
    model = joblib.load('outputs/rf_model.pkl')
    scores_ref = pd.read_csv('outputs/rf_scores.csv').iloc[0]
else:
    model = joblib.load('outputs/svm_model.pkl')
    scores_ref = pd.read_csv('outputs/svm_scores.csv').iloc[0]

print(f"  ✅ Model loaded successfully")


# 2. LOAD DATA

X_scaled    = np.load('outputs/X_scaled.npy')
y_resampled = np.load('outputs/y_resampled.npy')
le_target   = joblib.load('outputs/label_encoder.pkl')
class_names = le_target.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled,
    test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"\n  Train samples : {X_train.shape[0]}")
print(f"  Test  samples : {X_test.shape[0]}")


# 3. PREDICTIONS

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


# 4. METRICS

accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc     = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

print("\n" + "=" * 55)
print(f"   {best_model_name.upper()} — FINAL EVALUATION RESULTS")
print("=" * 55)
print(f"  Accuracy            : {accuracy*100:.2f}%")
print(f"  F1 Score (macro)    : {f1_macro:.4f}")
print(f"  F1 Score (weighted) : {f1_weighted:.4f}")
print(f"  ROC-AUC Score       : {roc_auc:.4f}")
print("\n  Per-Class Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# PLOT 1: Final Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5)
plt.title(f'{best_model_name} — Final Confusion Matrix',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/final_confusion_matrix.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/final_confusion_matrix.png")


# PLOT 2: ROC Curve (One vs Rest — per class)

y_test_bin = label_binarize(y_test, classes=np.unique(y_resampled))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

plt.figure(figsize=(9, 7))
for i, (cls, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{cls} (AUC = {roc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'{best_model_name} — ROC Curves (One vs Rest)',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/final_roc_curve.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/final_roc_curve.png")


# PLOT 3: Learning Curve

print("\n  Generating Learning Curve (may take a moment)...")
train_sizes, train_scores, val_scores = learning_curve(
    model, X_scaled, y_resampled,
    cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1) * 100
train_std  = train_scores.std(axis=1)  * 100
val_mean   = val_scores.mean(axis=1)   * 100
val_std    = val_scores.std(axis=1)    * 100

plt.figure(figsize=(9, 6))
plt.plot(train_sizes, train_mean, 'o-', color='steelblue',
         lw=2, label='Training Accuracy')
plt.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color='steelblue')
plt.plot(train_sizes, val_mean, 's-', color='darkorange',
         lw=2, label='Validation Accuracy')
plt.fill_between(train_sizes,
                 val_mean - val_std,
                 val_mean + val_std,
                 alpha=0.15, color='darkorange')
plt.xlabel('Training Samples', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'{best_model_name} — Learning Curve',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/final_learning_curve.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/final_learning_curve.png")


# PLOT 4: F1 Score per Class

report = classification_report(y_test, y_pred,
                                target_names=class_names,
                                output_dict=True)
f1_per_class = [report[cls]['f1-score'] for cls in class_names]

plt.figure(figsize=(9, 5))
bars = plt.bar(class_names, f1_per_class, color=colors, edgecolor='black')
for bar, val in zip(bars, f1_per_class):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
plt.title(f'{best_model_name} — F1 Score per Class',
          fontsize=14, fontweight='bold')
plt.xlabel('ECG Condition', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.ylim(0, 1.15)
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/final_f1_per_class.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/final_f1_per_class.png")


# 5. FINAL SUMMARY

summary = {
    'Best Model'      : best_model_name,
    'Accuracy (%)'    : round(accuracy * 100, 2),
    'F1 Macro'        : round(f1_macro, 4),
    'F1 Weighted'     : round(f1_weighted, 4),
    'ROC-AUC'         : round(roc_auc, 4),
}
pd.DataFrame([summary]).to_csv('outputs/final_summary.csv', index=False)
print("\n  ✅ Saved: outputs/final_summary.csv")

print("\n" + "=" * 55)
print(f"  ✅ FILE 6 COMPLETE!")
print(f"  🏆 Best Model  : {best_model_name}")
print(f"  📊 Accuracy    : {accuracy*100:.2f}%")
print(f"  📊 ROC-AUC     : {roc_auc:.4f}")
print(f"  📁 All plots saved in: outputs/")
print("=" * 55)