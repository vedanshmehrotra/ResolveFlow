"""
IEEE-style chart generation for Email Triage project.
Generates 4 figures saved to results/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results', exist_ok=True)

# ── Common style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       100,
})

# ════════════════════════════════════════════════════════════════════
# CHART 1 — Issue Model Comparison (Grouped Bar)
# ════════════════════════════════════════════════════════════════════
metrics      = ['Macro F1', 'Micro F1', 'Hamming Loss', 'Subset Accuracy']
lr_scores    = [0.9230,     0.9365,     0.0198,         0.8417]
lstm_scores  = [0.8440,     0.8776,     0.0375,         0.7250]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bar1 = ax.bar(x - width/2, lr_scores,   width, label='Logistic Regression', color='steelblue',  edgecolor='black', linewidth=0.7)
bar2 = ax.bar(x + width/2, lstm_scores, width, label='BiLSTM',              color='lightcoral', edgecolor='black', linewidth=0.7)

# Value labels
for bar in bar1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f'{h:.4f}',
            ha='center', va='bottom', fontsize=8)
for bar in bar2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f'{h:.4f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Issue Classifier: Logistic Regression vs BiLSTM')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('results/issue_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: issue_model_comparison.png")

# ════════════════════════════════════════════════════════════════════
# CHART 2 — Per-Class F1 (Horizontal Bar, LR)
# ════════════════════════════════════════════════════════════════════
categories = [
    'billing_issue',
    'cleanliness_issue',
    'electrical_issue',
    'food_issue',
    'furniture_issue',
    'internet_issue',
    'noise_issue',
    'plumbing_issue',
]
f1_scores = [1.0000, 0.6957, 0.8750, 0.9615, 0.9655, 0.9565, 0.9545, 0.9756]

# Sort ascending for readability
sorted_pairs = sorted(zip(f1_scores, categories))
f1_sorted    = [p[0] for p in sorted_pairs]
cats_sorted  = [p[1] for p in sorted_pairs]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(cats_sorted, f1_sorted, color='steelblue', edgecolor='black', linewidth=0.7)

for bar, val in zip(bars, f1_sorted):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

ax.set_xlabel('F1 Score')
ax.set_title('Issue Classifier (LR): Per-Class F1 Score')
ax.set_xlim(0, 1.12)
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('results/issue_f1_per_class.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: issue_f1_per_class.png")

# ════════════════════════════════════════════════════════════════════
# CHART 3 — Urgency Confusion Matrix (imshow heatmap)
# ════════════════════════════════════════════════════════════════════
cm_data = np.array([[36, 0,  0],
                    [ 0, 61, 1],
                    [ 0,  2, 20]])
labels  = ['Low', 'Medium', 'High']

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm_data, interpolation='nearest', cmap='Blues')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Urgency Classifier: Confusion Matrix (BiLSTM)')

thresh = cm_data.max() / 2.0
for i in range(3):
    for j in range(3):
        color = 'white' if cm_data[i, j] > thresh else 'black'
        ax.text(j, i, str(cm_data[i, j]), ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

fig.tight_layout()
fig.savefig('results/urgency_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: urgency_confusion_matrix.png")

# ════════════════════════════════════════════════════════════════════
# CHART 4 — Coverage–Risk Curve (full curve from CSV + key annotations)
# ════════════════════════════════════════════════════════════════════
import csv

thresholds_full, coverage_full, risk_full = [], [], []
with open('results/coverage_risk_curve.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cov  = float(row['coverage_pct'])
        risk = row['risk_pct']
        thresh = float(row['threshold'])
        if risk == '' or risk == 'NaN':
            continue
        thresholds_full.append(thresh)
        coverage_full.append(cov)
        risk_full.append(float(risk))

# Sort by coverage descending (high threshold → low coverage)
pairs = sorted(zip(coverage_full, risk_full, thresholds_full), reverse=True)
cov_plot  = [p[0] for p in pairs]
risk_plot = [p[1] for p in pairs]
thr_plot  = [p[2] for p in pairs]

# Key annotation points
key_points = {
    0.50: (100.0, 0.83),
    0.65: (86.7,  0.00),
    0.85: (23.3,  0.00),
}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(cov_plot, risk_plot, 'o-', color='steelblue', markersize=4,
        linewidth=1.2, label='Coverage–Risk tradeoff')

# Annotate key thresholds
offsets = {
    0.50: (2,  0.06),
    0.65: (1, -0.09),
    0.85: (1,  0.06),
}
for thr, (cov, risk) in key_points.items():
    dx, dy = offsets[thr]
    ax.plot(cov, risk, 'rs', markersize=7, zorder=5)
    ax.annotate(f't={thr:.2f}\n({cov:.1f}%, {risk:.2f}%)',
                xy=(cov, risk),
                xytext=(cov + dx, risk + dy),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=0.8),
                ha='left')

ax.set_xlabel('Coverage (%)')
ax.set_ylabel('Risk (%)')
ax.set_title('Selective Classification: Coverage vs. Risk Curve')
ax.set_xlim(-2, 108)
ax.set_ylim(-0.15, 1.2)
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('results/coverage_risk_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: coverage_risk_curve.png")

print("\nAll 4 charts generated successfully.")
