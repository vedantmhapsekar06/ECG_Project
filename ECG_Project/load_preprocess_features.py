
# FILE 2: Load Dataset + Preprocessing + Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

os.makedirs('outputs', exist_ok=True)


# 1. LOAD DATASET

print("=" * 55)
print("   STEP 1: LOADING DATASET")
print("=" * 55)

df = pd.read_csv('imbalanced_ecg_dataset.csv')

print(f"  Shape            : {df.shape}")
print(f"  Columns          : {df.columns.tolist()}")
print(f"  Missing Values   : {df.isnull().sum().sum()}")
print(f"  Duplicate Rows   : {df.duplicated().sum()}")
print(f"\n  Class Distribution (Before SMOTE):")
print(df['Condition'].value_counts().to_string())


# 2. PREPROCESSING

print("\n" + "=" * 55)
print("   STEP 2: PREPROCESSING")
print("=" * 55)

# Drop duplicates if any
df = df.drop_duplicates()

# Drop Patient_ID (not useful for ML)
df = df.drop(columns=['Patient_ID'])

# Encode Gender: Female=0, Male=1
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
print("  ✅ Gender encoded  : Female=0, Male=1")

# Encode target label
le_target = LabelEncoder()
df['Label_Encoded'] = le_target.fit_transform(df['Condition'])
class_names = le_target.classes_
print(f"  ✅ Target encoded  : {dict(enumerate(class_names))}")


# 3. FEATURE ENGINEERING

print("\n" + "=" * 55)
print("   STEP 3: FEATURE ENGINEERING")
print("=" * 55)

# New Feature 1: QT Corrected Ratio
df['QT_QTc_Ratio'] = df['QT_Interval_ms'] / (df['QTc_ms'] + 1e-5)

# New Feature 2: PR to QRS Ratio
df['PR_QRS_Ratio'] = df['PR_Interval_ms'] / (df['QRS_Duration_ms'] + 1e-5)

# New Feature 3: Wave Amplitude Ratio
df['PRT_Amplitude_Sum'] = (df['P_Wave_Amplitude_mV'] +
                            df['R_Wave_Amplitude_mV'] +
                            df['T_Wave_Amplitude_mV'])

# New Feature 4: Pulse Pressure
df['Pulse_Pressure'] = df['Systolic_BP_mmHg'] - df['Diastolic_BP_mmHg']

# New Feature 5: HRV to Heart Rate Ratio
df['HRV_HR_Ratio'] = df['HRV_SDNN_ms'] / (df['Heart_Rate_BPM'] + 1e-5)

print("  ✅ New features created:")
print("     - QT_QTc_Ratio")
print("     - PR_QRS_Ratio")
print("     - PRT_Amplitude_Sum")
print("     - Pulse_Pressure")
print("     - HRV_HR_Ratio")


# 4. DEFINE FEATURES & TARGET

X = df.drop(columns=['Condition', 'Label', 'Label_Encoded'])
y = df['Label_Encoded']

print(f"\n  Total Features   : {X.shape[1]}")
print(f"  Feature List     : {X.columns.tolist()}")


# 5. HANDLE CLASS IMBALANCE — SMOTE

print("\n" + "=" * 55)
print("   STEP 4: HANDLING IMBALANCE WITH SMOTE")
print("=" * 55)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"  Before SMOTE: {dict(zip(class_names, np.bincount(y)))}")
print(f"  After  SMOTE: {dict(zip(class_names, np.bincount(y_resampled)))}")


# 6. FEATURE SCALING

print("\n" + "=" * 55)
print("   STEP 5: FEATURE SCALING")
print("=" * 55)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
print("  ✅ StandardScaler applied")


# 7. PLOT — Class distribution Before vs After SMOTE

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Before
before_counts = pd.Series(y.values).map(dict(enumerate(class_names))).value_counts()
axes[0].bar(before_counts.index, before_counts.values,
            color=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'],
            edgecolor='black')
axes[0].set_title('Class Distribution — BEFORE SMOTE', fontweight='bold')
axes[0].set_xlabel('Condition')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(before_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# After
after_counts = pd.Series(y_resampled).map(dict(enumerate(class_names))).value_counts()
axes[1].bar(after_counts.index, after_counts.values,
            color=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'],
            edgecolor='black')
axes[1].set_title('Class Distribution — AFTER SMOTE', fontweight='bold')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=20)
for i, v in enumerate(after_counts.values):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.suptitle('SMOTE — Before vs After Balancing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/smote_balancing.png', dpi=150)
plt.show()
print("\n  ✅ Saved: outputs/smote_balancing.png")


# 8. SAVE PROCESSED DATA

np.save('outputs/X_scaled.npy', X_scaled)
np.save('outputs/y_resampled.npy', y_resampled)
joblib.dump(scaler,    'outputs/scaler.pkl')
joblib.dump(le_target, 'outputs/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'outputs/feature_names.pkl')

print("\n  ✅ Saved: outputs/X_scaled.npy")
print("  ✅ Saved: outputs/y_resampled.npy")
print("  ✅ Saved: outputs/scaler.pkl")
print("  ✅ Saved: outputs/label_encoder.pkl")

print("\n" + "=" * 55)
print("  ✅ FILE 2 COMPLETE!")
print("  ▶️  Next: Run 3_ml_model1_random_forest.py")
print("=" * 55)