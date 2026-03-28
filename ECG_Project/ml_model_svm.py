
# FILE 4: ML Model 2 — Support Vector Machine (SVM)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

os.makedirs('outputs', exist_ok=True)


# 1. LOAD PREPROCESSED DATA

X_scaled    = np.load('outputs/X_scaled.npy')
y_resampled = np.load('outputs/y_resampled.npy')
le_target   = joblib.load('outputs/label_encoder.pkl')
class_names = le_target.classes_

print("=" * 55)
print("   FILE 4: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 55)
print(f"  Data shape  : {X_scaled.shape}")
print(f"  Classes     : {class_names}")


# 2. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled,
    test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"\n  Train samples : {X_train.shape[0]}")
print(f"  Test  samples : {X_test.shape[0]}")


# 3. TRAIN SVM

print("\n  Training SVM (may take a moment)...")
svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)
print("  ✅ Training Complete!")


# 4. PREDICTIONS

y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)


# 5. METRICS

accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc     = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_scaled, y_resampled, cv=skf, scoring='accuracy')

print("\n" + "=" * 55)
print("   SVM — RESULTS")
print("=" * 55)
print(f"  Accuracy            : {accuracy*100:.2f}%")
print(f"  F1 Score (macro)    : {f1_macro:.4f}")
print(f"  F1 Score (weighted) : {f1_weighted:.4f}")
print(f"  ROC-AUC Score       : {roc_auc:.4f}")
print(f"  CV Accuracy (5-Fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# 6. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM — Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('outputs/svm_confusion_matrix.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/svm_confusion_matrix.png")


# 7. SAVE MODEL & SCORES

joblib.dump(svm, 'outputs/svm_model.pkl')

scores = {
    'model'        : 'SVM',
    'accuracy'     : round(accuracy * 100, 2),
    'f1_macro'     : round(f1_macro, 4),
    'f1_weighted'  : round(f1_weighted, 4),
    'roc_auc'      : round(roc_auc, 4),
    'cv_mean'      : round(cv_scores.mean() * 100, 2),
    'cv_std'       : round(cv_scores.std() * 100, 2)
}
pd.DataFrame([scores]).to_csv('outputs/svm_scores.csv', index=False)

print("\n  ✅ Saved: outputs/svm_model.pkl")
print("  ✅ Saved: outputs/svm_scores.csv")
print("\n" + "=" * 55)
print("  ✅ FILE 4 COMPLETE!")
print("  ▶️  Next: Run 5_compare.py")
print("=" * 55)