"""
HOSTEL COMPLAINT TRIAGE SYSTEM - PHASE 3: ML BASELINE MODELS
=============================================================
Author: Vedansh
Date: December 2025

This script trains traditional ML models:
1. Issue Classifier: Logistic Regression (multi-label)
2. Urgency Classifier: Naive Bayes (multi-class)
3. Evaluation and comparison

Run this after Phase 2 (feature_engineering.py)
"""

import numpy as np
import json
import pickle
import os
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report, 
    hamming_loss, 
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("PHASE 3: MACHINE LEARNING BASELINE MODELS")
print("="*80)

#==============================================================================
# STEP 1: LOAD DATA AND FEATURES
#==============================================================================

print("\n" + "-"*80)
print("LOADING DATA")
print("-"*80)

try:
    # Load TF-IDF features
    X_train_tfidf = sparse.load_npz('data/X_train_tfidf.npz')
    X_test_tfidf = sparse.load_npz('data/X_test_tfidf.npz')
    
    # Load labels
    y_issues_train = np.load('data/y_issues_train.npy')
    y_issues_test = np.load('data/y_issues_test.npy')
    y_urgency_train = np.load('data/y_urgency_train.npy')
    y_urgency_test = np.load('data/y_urgency_test.npy')
    
    # Load category mappings
    with open('data/category_mapping.json', 'r') as f:
        mappings = json.load(f)
        categories = mappings['categories']
        urgency_mapping = mappings['urgency_mapping']
    
    print(f"\n✓ Data loaded successfully:")
    print(f"  Features: {X_train_tfidf.shape[1]}")
    print(f"  Training samples: {X_train_tfidf.shape[0]}")
    print(f"  Test samples: {X_test_tfidf.shape[0]}")
    print(f"  Issue categories: {len(categories)}")
    print(f"  Urgency levels: {len(urgency_mapping)}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required data files not found!")
    print(f"   Missing: {e.filename}")
    print(f"\n   Please run phase2_feature_engineering.py first!")
    exit(1)

#==============================================================================
# STEP 2: TRAIN ISSUE CLASSIFIER (Multi-label)
#==============================================================================

print("\n" + "="*80)
print("MODEL 1: ISSUE TYPE CLASSIFIER (Multi-label)")
print("="*80)

print("\nAlgorithm: Logistic Regression with One-vs-Rest")
print("Task: Predict one or more issue categories from 8 options")

# Initialize model
print("\nTraining model...")
start_time = time.time()

issue_classifier = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        solver='lbfgs'
    ),
    n_jobs=-1  # Use all CPU cores
)

# Train
issue_classifier.fit(X_train_tfidf, y_issues_train)

train_time = time.time() - start_time
print(f"✓ Training completed in {train_time:.2f} seconds")

# Predictions
print("\nMaking predictions...")
y_issues_pred_train = issue_classifier.predict(X_train_tfidf)
y_issues_pred_test = issue_classifier.predict(X_test_tfidf)

# Get prediction probabilities (returns list of arrays for OneVsRest)
print("Calculating prediction probabilities...")
y_issues_proba_test = issue_classifier.predict_proba(X_test_tfidf)

#==============================================================================
# STEP 3: EVALUATE ISSUE CLASSIFIER
#==============================================================================

print("\n" + "-"*80)
print("ISSUE CLASSIFIER EVALUATION")
print("-"*80)

# Calculate metrics
train_hamming = hamming_loss(y_issues_train, y_issues_pred_train)
test_hamming = hamming_loss(y_issues_test, y_issues_pred_test)

train_f1_micro = f1_score(y_issues_train, y_issues_pred_train, average='micro')
test_f1_micro = f1_score(y_issues_test, y_issues_pred_test, average='micro')

train_f1_macro = f1_score(y_issues_train, y_issues_pred_train, average='macro')
test_f1_macro = f1_score(y_issues_test, y_issues_pred_test, average='macro')

# Subset accuracy (exact match)
train_subset_acc = accuracy_score(y_issues_train, y_issues_pred_train)
test_subset_acc = accuracy_score(y_issues_test, y_issues_pred_test)

print("\nOverall Metrics:")
print(f"{'Metric':<25s} {'Training':<15s} {'Test':<15s}")
print("-" * 55)
print(f"{'Hamming Loss':<25s} {train_hamming:<15.4f} {test_hamming:<15.4f}")
print(f"{'Subset Accuracy':<25s} {train_subset_acc:<15.4f} {test_subset_acc:<15.4f}")
print(f"{'F1-Score (Micro)':<25s} {train_f1_micro:<15.4f} {test_f1_micro:<15.4f}")
print(f"{'F1-Score (Macro)':<25s} {train_f1_macro:<15.4f} {test_f1_macro:<15.4f}")

# Per-class metrics - FIXED VERSION
print("\n" + "-"*80)
print("PER-CATEGORY PERFORMANCE (Test Set)")
print("-"*80)

print(f"\n{'Category':<25s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s} {'Support':<10s}")
print("-" * 71)

f1_scores_per_cat = []
for i, category in enumerate(categories):
    y_true_cat = y_issues_test[:, i]
    y_pred_cat = y_issues_pred_test[:, i]
    
    # Calculate metrics for this category
    # Use average=None to get per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, average=None, zero_division=0
    )
    
    # Extract positive class metrics (index 1 = issue present)
    # Handle case where only one class is present
    if len(precision) > 1:
        precision_pos = precision[1]
        recall_pos = recall[1]
        f1_pos = f1[1]
        support_pos = support[1]
    else:
        # Only one class present in test set
        precision_pos = precision[0]
        recall_pos = recall[0]
        f1_pos = f1[0]
        support_pos = support[0]
    
    f1_scores_per_cat.append(f1_pos)
    print(f"{category:<25s} {precision_pos:<12.3f} {recall_pos:<12.3f} {f1_pos:<12.3f} {int(support_pos):<10d}")

#==============================================================================
# STEP 4: TRAIN URGENCY CLASSIFIER (Multi-class)
#==============================================================================

print("\n" + "="*80)
print("MODEL 2: URGENCY CLASSIFIER (Multi-class)")
print("="*80)

print("\nAlgorithm: Multinomial Naive Bayes")
print("Task: Predict urgency level (low, medium, high)")

# Initialize model
print("\nTraining model...")
start_time = time.time()

urgency_classifier = MultinomialNB(alpha=0.1)  # Laplace smoothing

# Train
urgency_classifier.fit(X_train_tfidf, y_urgency_train)

train_time = time.time() - start_time
print(f"✓ Training completed in {train_time:.2f} seconds")

# Predictions
print("\nMaking predictions...")
y_urgency_pred_train = urgency_classifier.predict(X_train_tfidf)
y_urgency_pred_test = urgency_classifier.predict(X_test_tfidf)

# Get prediction probabilities
y_urgency_proba_test = urgency_classifier.predict_proba(X_test_tfidf)

#==============================================================================
# STEP 5: EVALUATE URGENCY CLASSIFIER
#==============================================================================

print("\n" + "-"*80)
print("URGENCY CLASSIFIER EVALUATION")
print("-"*80)

# Calculate metrics
train_acc = accuracy_score(y_urgency_train, y_urgency_pred_train)
test_acc = accuracy_score(y_urgency_test, y_urgency_pred_test)

train_f1 = f1_score(y_urgency_train, y_urgency_pred_train, average='weighted')
test_f1 = f1_score(y_urgency_test, y_urgency_pred_test, average='weighted')

print("\nOverall Metrics:")
print(f"{'Metric':<25s} {'Training':<15s} {'Test':<15s}")
print("-" * 55)
print(f"{'Accuracy':<25s} {train_acc:<15.4f} {test_acc:<15.4f}")
print(f"{'F1-Score (Weighted)':<25s} {train_f1:<15.4f} {test_f1:<15.4f}")

# Classification report
print("\n" + "-"*80)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("-"*80)

urgency_labels = ['low', 'medium', 'high']
print("\n" + classification_report(
    y_urgency_test, 
    y_urgency_pred_test,
    target_names=urgency_labels,
    digits=3
))

# Confusion matrix
cm = confusion_matrix(y_urgency_test, y_urgency_pred_test)
print("\nConfusion Matrix:")
print(f"{'':8s}", end='')
for label in urgency_labels:
    print(f"{label:>10s}", end='')
print()
for i, label in enumerate(urgency_labels):
    print(f"{label:8s}", end='')
    for j in range(len(urgency_labels)):
        print(f"{cm[i,j]:10d}", end='')
    print()

#==============================================================================
# STEP 6: VISUALIZE RESULTS
#==============================================================================

print("\n" + "-"*80)
print("GENERATING VISUALIZATIONS")
print("-"*80)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Issue Classifier - Per-category F1 scores
ax1 = plt.subplot(2, 3, 1)
bars = ax1.barh(categories, f1_scores_per_cat, color='steelblue')
ax1.set_xlabel('F1-Score')
ax1.set_title('Issue Classifier: F1-Score per Category', fontweight='bold')
ax1.set_xlim([0, 1])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}', va='center', fontsize=9)

# Plot 2: Issue Classifier - Overall metrics comparison
ax2 = plt.subplot(2, 3, 2)
metrics_names = ['Hamming\nLoss', 'Subset\nAccuracy', 'F1-Micro', 'F1-Macro']
train_metrics = [train_hamming, train_subset_acc, train_f1_micro, train_f1_macro]
test_metrics = [test_hamming, test_subset_acc, test_f1_micro, test_f1_macro]

x = np.arange(len(metrics_names))
width = 0.35
ax2.bar(x - width/2, train_metrics, width, label='Train', color='lightblue')
ax2.bar(x + width/2, test_metrics, width, label='Test', color='darkblue')
ax2.set_ylabel('Score')
ax2.set_title('Issue Classifier: Overall Metrics', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_names)
ax2.legend()
ax2.set_ylim([0, 1])

# Plot 3: Urgency Classifier - Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=urgency_labels, yticklabels=urgency_labels,
            ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title('Urgency Classifier: Confusion Matrix', fontweight='bold')

# Plot 4: Urgency Classifier - Per-class metrics
ax4 = plt.subplot(2, 3, 4)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_urgency_test, y_urgency_pred_test, average=None
)
x = np.arange(len(urgency_labels))
width = 0.25
ax4.bar(x - width, precision, width, label='Precision', color='green', alpha=0.7)
ax4.bar(x, recall, width, label='Recall', color='orange', alpha=0.7)
ax4.bar(x + width, f1, width, label='F1-Score', color='red', alpha=0.7)
ax4.set_ylabel('Score')
ax4.set_title('Urgency Classifier: Per-Class Metrics', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(urgency_labels)
ax4.legend()
ax4.set_ylim([0, 1])

# Plot 5: Model comparison (accuracy/F1)
ax5 = plt.subplot(2, 3, 5)
models = ['Issue\n(F1-Macro)', 'Urgency\n(Accuracy)']
train_scores = [train_f1_macro, train_acc]
test_scores = [test_f1_macro, test_acc]
x = np.arange(len(models))
width = 0.35
ax5.bar(x - width/2, train_scores, width, label='Train', color='lightgreen')
ax5.bar(x + width/2, test_scores, width, label='Test', color='darkgreen')
ax5.set_ylabel('Score')
ax5.set_title('Model Comparison: Train vs Test', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(models)
ax5.legend()
ax5.set_ylim([0, 1])

# Plot 6: Sample predictions confidence
ax6 = plt.subplot(2, 3, 6)
# Show confidence distribution for correct vs incorrect predictions
correct_mask = (y_urgency_pred_test == y_urgency_test)
correct_confidences = y_urgency_proba_test[correct_mask].max(axis=1)
incorrect_confidences = y_urgency_proba_test[~correct_mask].max(axis=1)

ax6.hist([correct_confidences, incorrect_confidences], 
         bins=20, label=['Correct', 'Incorrect'], 
         color=['green', 'red'], alpha=0.6)
ax6.set_xlabel('Prediction Confidence')
ax6.set_ylabel('Count')
ax6.set_title('Urgency: Prediction Confidence Distribution', fontweight='bold')
ax6.legend()

plt.tight_layout()
plt.savefig('results/ml_baseline_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'results/ml_baseline_results.png'")

#==============================================================================
# STEP 7: SAVE MODELS
#==============================================================================

print("\n" + "-"*80)
print("SAVING MODELS")
print("-"*80)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save models
with open('models/issue_classifier_ml.pkl', 'wb') as f:
    pickle.dump(issue_classifier, f)

with open('models/urgency_classifier_ml.pkl', 'wb') as f:
    pickle.dump(urgency_classifier, f)

# Save performance metrics
results = {
    'issue_classifier': {
        'train_hamming_loss': float(train_hamming),
        'test_hamming_loss': float(test_hamming),
        'train_f1_micro': float(train_f1_micro),
        'test_f1_micro': float(test_f1_micro),
        'train_f1_macro': float(train_f1_macro),
        'test_f1_macro': float(test_f1_macro),
        'train_subset_accuracy': float(train_subset_acc),
        'test_subset_accuracy': float(test_subset_acc),
        'per_category_f1': {cat: float(score) for cat, score in zip(categories, f1_scores_per_cat)}
    },
    'urgency_classifier': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_f1_weighted': float(train_f1),
        'test_f1_weighted': float(test_f1),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist()
    }
}

with open('results/ml_baseline_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Models and metrics saved:")
print("  - models/issue_classifier_ml.pkl")
print("  - models/urgency_classifier_ml.pkl")
print("  - results/ml_baseline_metrics.json")

#==============================================================================
# STEP 8: EXAMPLE PREDICTIONS
#==============================================================================

print("\n" + "-"*80)
print("EXAMPLE PREDICTIONS")
print("-"*80)

# Load original text for examples
X_test_original = np.load('data/X_test.npy', allow_pickle=True)

print("\nShowing 3 random test examples:\n")

# Select 3 random indices
np.random.seed(42)
example_indices = np.random.choice(len(X_test_original), 3, replace=False)

for idx in example_indices:
    print("-" * 80)
    complaint_text = X_test_original[idx][:200] + "..."
    
    # Actual labels
    actual_issues = [categories[i] for i in range(len(categories)) if y_issues_test[idx, i] == 1]
    actual_urgency = urgency_labels[y_urgency_test[idx]]
    
    # Predicted labels
    pred_issues = [categories[i] for i in range(len(categories)) if y_issues_pred_test[idx, i] == 1]
    pred_urgency = urgency_labels[y_urgency_pred_test[idx]]
    pred_urgency_conf = y_urgency_proba_test[idx].max()
    
    print(f"Complaint: {complaint_text}")
    print(f"\nActual:")
    print(f"  Issues:  {', '.join(actual_issues)}")
    print(f"  Urgency: {actual_urgency}")
    print(f"\nPredicted:")
    print(f"  Issues:  {', '.join(pred_issues)}")
    print(f"  Urgency: {pred_urgency} (confidence: {pred_urgency_conf:.2f})")
    print()

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "="*80)
print("PHASE 3 COMPLETE - ML BASELINE MODELS SUMMARY")
print("="*80)
print(f"""
Issue Classifier (Logistic Regression - Multi-label):
  • Algorithm: One-vs-Rest Logistic Regression
  • Test Hamming Loss: {test_hamming:.4f}
  • Test F1-Score (Macro): {test_f1_macro:.4f}
  • Test Subset Accuracy: {test_subset_acc:.4f}
  
Urgency Classifier (Naive Bayes - Multi-class):
  • Algorithm: Multinomial Naive Bayes
  • Test Accuracy: {test_acc:.4f}
  • Test F1-Score (Weighted): {test_f1:.4f}

✅ These are realistic baseline scores for synthetic data!
   Lower scores expected vs production systems with real data.

📊 Results saved to:
   • results/ml_baseline_results.png (visualizations)
   • results/ml_baseline_metrics.json (metrics)
   • models/ (trained models)

Next Steps:
  → Phase 4: Train Deep Learning Models (LSTM/BERT)
     Expected improvement: 5-10% over baseline
     Better handling of context and semantic meaning
  
  → Create inference script for real-time predictions
  → Build deployment pipeline with confidence routing
""")

print("="*80)
print("✅ Phase 3 Complete! Ready for Phase 4 🚀")
print("="*80)