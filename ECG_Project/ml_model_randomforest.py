
# FILE 3: ML Model 1 — Random Forest Classifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

os.makedirs('outputs', exist_ok=True)


# 1. LOAD PREPROCESSED DATA

X_scaled    = np.load('outputs/X_scaled.npy')
y_resampled = np.load('outputs/y_resampled.npy')
le_target   = joblib.load('outputs/label_encoder.pkl')
class_names = le_target.classes_

print("=" * 55)
print("   FILE 3: RANDOM FOREST CLASSIFIER")
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


# 3. TRAIN RANDOM FOREST

print("\n  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("  ✅ Training Complete!")


# 4. PREDICTIONS

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)


# 5. METRICS

accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc     = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y_resampled, cv=skf, scoring='accuracy')

print("\n" + "=" * 55)
print("   RANDOM FOREST — RESULTS")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Random Forest — Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('outputs/rf_confusion_matrix.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/rf_confusion_matrix.png")


# 7. FEATURE IMPORTANCE

feature_names = joblib.load('outputs/feature_names.pkl')
feat_imp = pd.Series(rf.feature_importances_,
                     index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(11, 6))
feat_imp.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig('outputs/rf_feature_importance.png', dpi=150)
plt.show()
print("  ✅ Saved: outputs/rf_feature_importance.png")


# 8. SAVE MODEL & SCORES

joblib.dump(rf, 'outputs/rf_model.pkl')

scores = {
    'model'        : 'Random Forest',
    'accuracy'     : round(accuracy * 100, 2),
    'f1_macro'     : round(f1_macro, 4),
    'f1_weighted'  : round(f1_weighted, 4),
    'roc_auc'      : round(roc_auc, 4),
    'cv_mean'      : round(cv_scores.mean() * 100, 2),
    'cv_std'       : round(cv_scores.std() * 100, 2)
}
pd.DataFrame([scores]).to_csv('outputs/rf_scores.csv', index=False)

print("\n  ✅ Saved: outputs/rf_model.pkl")
print("  ✅ Saved: outputs/rf_scores.csv")
print("\n" + "=" * 55)
print("  ✅ FILE 3 COMPLETE!")
print("  ▶️  Next: Run 4_ml_model2_svm.py")
print("=" * 55)