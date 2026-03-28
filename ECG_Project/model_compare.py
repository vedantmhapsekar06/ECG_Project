
# FILE 5: Compare Random Forest vs SVM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)


# 1. LOAD SCORES

rf_scores  = pd.read_csv('outputs/rf_scores.csv').iloc[0]
svm_scores = pd.read_csv('outputs/svm_scores.csv').iloc[0]

metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'cv_mean']
labels  = ['Accuracy (%)', 'F1 Macro', 'F1 Weighted', 'ROC-AUC', 'CV Accuracy (%)']

rf_vals  = [rf_scores[m]  for m in metrics]
svm_vals = [svm_scores[m] for m in metrics]


# 2. COMPARISON TABLE

print("=" * 62)
print("        RANDOM FOREST  vs  SVM — COMPARISON TABLE")
print("=" * 62)
print(f"{'Metric':<22} {'Random Forest':>15} {'SVM':>12}  {'Winner':>8}")
print("-" * 62)

rf_wins = 0
for label, m, rv, sv in zip(labels, metrics, rf_vals, svm_vals):
    if rv >= sv:
        winner = '🌲 RF'
        rf_wins += 1
    else:
        winner = '⚡ SVM'
    print(f"  {label:<20} {rv:>15.2f} {sv:>12.2f}  {winner:>8}")

svm_wins = len(metrics) - rf_wins
print("=" * 62)
print(f"\n  🌲 Random Forest wins : {rf_wins}/{len(metrics)}")
print(f"  ⚡ SVM wins           : {svm_wins}/{len(metrics)}")

# Determine best
if rf_wins > svm_wins:
    best = 'Random Forest'
elif svm_wins > rf_wins:
    best = 'SVM'
else:
    best = 'Random Forest' if rf_scores['roc_auc'] >= svm_scores['roc_auc'] else 'SVM'
    print(f"\n  🤝 Tie — decided by ROC-AUC")

print(f"\n  🏆 BEST MODEL  →  {best.upper()}")
print("=" * 62)


# 3. BAR CHART

x     = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, rf_vals,  width, label='🌲 Random Forest',
               color='steelblue',  edgecolor='black')
bars2 = ax.bar(x + width/2, svm_vals, width, label='⚡ SVM',
               color='darkorange', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='steelblue')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='darkorange')

ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Random Forest vs SVM — Performance Comparison\n🏆 Best Model: {best}',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(rf_vals), max(svm_vals)) * 1.18)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/comparison_bar_chart.png', dpi=150)
plt.show()
print("\n  ✅ Saved: outputs/comparison_bar_chart.png")


# 4. RADAR CHART

radar_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'ROC-AUC', 'CV Accuracy']
rf_radar     = [rf_scores['accuracy']/100,  rf_scores['f1_macro'],
                rf_scores['f1_weighted'],   rf_scores['roc_auc'],
                rf_scores['cv_mean']/100]
svm_radar    = [svm_scores['accuracy']/100, svm_scores['f1_macro'],
                svm_scores['f1_weighted'],  svm_scores['roc_auc'],
                svm_scores['cv_mean']/100]

angles    = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
rf_radar  += rf_radar[:1]
svm_radar += svm_radar[:1]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, rf_radar,  'o-', lw=2, color='steelblue',  label='🌲 Random Forest')
ax.fill(angles, rf_radar,  alpha=0.2, color='steelblue')
ax.plot(angles, svm_radar, 's-', lw=2, color='darkorange', label='⚡ SVM')
ax.fill(angles, svm_radar, alpha=0.2, color='darkorange')
ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title(f'RF vs SVM — Radar Chart\n🏆 Best: {best}',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
plt.tight_layout()
plt.savefig('outputs/comparison_radar_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Saved: outputs/comparison_radar_chart.png")


# 5. SAVE BEST MODEL NAME FOR FILE 6

pd.DataFrame([{'best_model': best}]).to_csv('outputs/best_model.csv', index=False)
print("  ✅ Saved: outputs/best_model.csv")

print("\n" + "=" * 62)
print("  ✅ FILE 5 COMPLETE!")
print(f"  🏆 Best Model saved : {best}")
print("  ▶️  Next: Run 6_model_evaluation.py")
print("=" * 62)
