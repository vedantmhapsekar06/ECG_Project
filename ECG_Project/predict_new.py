# FILE 7: Predict Conditions for New Patients
# Uses the best trained model (chosen in File 5) to predict
# cardiac conditions for brand new unseen patient data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)


# 1. LOAD BEST MODEL & TOOLS

print("=" * 60)
print("   FILE 7: PREDICT NEW PATIENTS")
print("=" * 60)

best_model_name = pd.read_csv('outputs/best_model.csv').iloc[0]['best_model']
print(f"\n  Best Model       : {best_model_name}")

if best_model_name == 'Random Forest':
    model = joblib.load('outputs/rf_model.pkl')
else:
    model = joblib.load('outputs/svm_model.pkl')

scaler        = joblib.load('outputs/scaler.pkl')
le_target     = joblib.load('outputs/label_encoder.pkl')
feature_names = joblib.load('outputs/feature_names.pkl')
class_names   = le_target.classes_

print(f"  Model loaded successfully")
print(f"  Classes          : {list(class_names)}")


# 2. LOAD NEW PATIENT DATA

print("\n" + "=" * 60)
print("   STEP 1: LOADING NEW PATIENT DATA")
print("=" * 60)

df_new = pd.read_csv('new_patients.csv')
print(f"\n  New patients loaded : {len(df_new)} records")


# 3. PREPROCESS (same steps as training)

print("\n" + "=" * 60)
print("   STEP 2: PREPROCESSING NEW DATA")
print("=" * 60)

# Store Patient IDs then drop
patient_ids = df_new['Patient_ID'].values
df_new = df_new.drop(columns=['Patient_ID'])

# Encode Gender
df_new['Gender'] = df_new['Gender'].map({'Female': 0, 'Male': 1})
print("  Gender encoded")

# Feature Engineering (same 5 features as training)
df_new['QT_QTc_Ratio']      = df_new['QT_Interval_ms'] / (df_new['QTc_ms'] + 1e-5)
df_new['PR_QRS_Ratio']      = df_new['PR_Interval_ms'] / (df_new['QRS_Duration_ms'] + 1e-5)
df_new['PRT_Amplitude_Sum'] = (df_new['P_Wave_Amplitude_mV'] +
                                df_new['R_Wave_Amplitude_mV'] +
                                df_new['T_Wave_Amplitude_mV'])
df_new['Pulse_Pressure']    = df_new['Systolic_BP_mmHg'] - df_new['Diastolic_BP_mmHg']
df_new['HRV_HR_Ratio']      = df_new['HRV_SDNN_ms'] / (df_new['Heart_Rate_BPM'] + 1e-5)
print("  Feature engineering applied")

# Reorder columns to match training
df_new = df_new[feature_names]

# Scale
X_new = scaler.transform(df_new)
print("  Scaling applied")


# 4. PREDICT

print("\n" + "=" * 60)
print("   STEP 3: PREDICTING CONDITIONS")
print("=" * 60)

y_pred      = model.predict(X_new)
y_prob      = model.predict_proba(X_new)
pred_labels = le_target.inverse_transform(y_pred)
confidence  = (y_prob.max(axis=1) * 100).round(2)

results = pd.DataFrame({
    'Patient_ID'          : patient_ids,
    'Predicted_Condition' : pred_labels,
    'Confidence_%'        : confidence,
})

print("\n  Prediction Results:")
print("-" * 60)
for _, row in results.iterrows():
    print(f"  Patient {int(row['Patient_ID']):>4}  |  "
          f"{row['Predicted_Condition']:<25}  |  "
          f"Confidence: {row['Confidence_%']}%")
print("-" * 60)


# 5. SAVE RESULTS

results.to_csv('outputs/new_patient_predictions.csv', index=False)
print("\n  Saved: outputs/new_patient_predictions.csv")


# 6. PLOT — Confidence Bar Chart

colors_map = {
    'Arrhythmia'           : '#e74c3c',
    'Heart Failure'        : '#3498db',
    'Myocardial Infarction': '#2ecc71',
    'Normal'               : '#f39c12',
    'Stress'               : '#9b59b6'
}

bar_colors = [colors_map.get(c, '#888') for c in results['Predicted_Condition']]

plt.figure(figsize=(10, 6))
bars = plt.barh(
    [f"Patient {int(pid)}" for pid in results['Patient_ID']],
    results['Confidence_%'],
    color=bar_colors, edgecolor='black'
)
for bar, val in zip(bars, results['Confidence_%']):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val}%', va='center', fontsize=9, fontweight='bold')

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
plt.legend(handles=legend_patches, fontsize=9, loc='lower right')
plt.xlabel('Confidence (%)', fontsize=11)
plt.title(f'{best_model_name} — Prediction Confidence for New Patients',
          fontsize=13, fontweight='bold')
plt.xlim(0, 115)
plt.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/new_patient_prediction_results.png', dpi=150)
plt.show()
print("  Saved: outputs/new_patient_prediction_results.png")


print("\n" + "=" * 60)
print(f"  FILE 7 COMPLETE!")
print(f"  Best Model          : {best_model_name}")
print(f"  New Patients Tested : {len(results)}")
print(f"  All outputs saved in: outputs/")
print("=" * 60)