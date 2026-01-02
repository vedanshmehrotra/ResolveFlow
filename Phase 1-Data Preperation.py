"""
HOSTEL COMPLAINT TRIAGE SYSTEM - PHASE 1: DATA PREPARATION
============================================================
Author: Vedansh
Date: December 2025

This script handles:
1. Loading validated dataset
2. Exploratory Data Analysis (EDA)
3. Train-test split
4. Data preprocessing
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#==============================================================================
# STEP 1: LOAD DATASET
#==============================================================================

print("="*80)
print("PHASE 1: DATA PREPARATION & EXPLORATORY ANALYSIS")
print("="*80)

# Load validated dataset
with open('hostel_dataset_validated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"\n✓ Dataset loaded successfully: {len(df)} entries\n")

# Display sample
print("Sample entries:")
print(df.head(3))
print("\n")

#==============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
#==============================================================================

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Basic statistics
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Text length statistics
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\n" + "-"*80)
print("TEXT STATISTICS")
print("-"*80)
print(f"Average complaint length: {df['text_length'].mean():.0f} characters")
print(f"Average word count: {df['word_count'].mean():.0f} words")
print(f"Min words: {df['word_count'].min()}")
print(f"Max words: {df['word_count'].max()}")

# Urgency distribution
print("\n" + "-"*80)
print("URGENCY DISTRIBUTION")
print("-"*80)
urgency_counts = df['urgency'].value_counts()
print(urgency_counts)
print(f"\nPercentages:")
print((urgency_counts / len(df) * 100).round(1))

# Issue category distribution
print("\n" + "-"*80)
print("ISSUE CATEGORY DISTRIBUTION")
print("-"*80)

# Flatten all labels to count individual categories
all_labels = []
for labels in df['labels']:
    all_labels.extend(labels)

label_counts = Counter(all_labels)
print("\nCategory frequencies:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label:25s}: {count:3d} ({count/len(df)*100:5.1f}%)")

# Multi-label statistics
df['num_labels'] = df['labels'].apply(len)
print("\n" + "-"*80)
print("MULTI-LABEL STATISTICS")
print("-"*80)
print(f"Single-issue complaints: {(df['num_labels']==1).sum()} ({(df['num_labels']==1).sum()/len(df)*100:.1f}%)")
print(f"Multi-issue complaints:  {(df['num_labels']>1).sum()} ({(df['num_labels']>1).sum()/len(df)*100:.1f}%)")
print(f"Max labels per complaint: {df['num_labels'].max()}")

#==============================================================================
# STEP 3: VISUALIZATIONS
#==============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Urgency Distribution
urgency_order = ['low', 'medium', 'high']
urgency_counts_ordered = df['urgency'].value_counts().reindex(urgency_order)
axes[0, 0].bar(urgency_order, urgency_counts_ordered, color=['green', 'orange', 'red'])
axes[0, 0].set_title('Urgency Level Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xlabel('Urgency Level')
for i, v in enumerate(urgency_counts_ordered):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Issue Category Distribution
categories = sorted(label_counts.keys())
counts = [label_counts[cat] for cat in categories]
axes[0, 1].barh(categories, counts, color='steelblue')
axes[0, 1].set_title('Issue Category Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Count')
for i, v in enumerate(counts):
    axes[0, 1].text(v + 2, i, str(v), va='center')

# Plot 3: Text Length Distribution
axes[1, 0].hist(df['word_count'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Complaint Length Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['word_count'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["word_count"].mean():.0f}')
axes[1, 0].legend()

# Plot 4: Single vs Multi-label
label_dist = df['num_labels'].value_counts().sort_index()
axes[1, 1].bar(label_dist.index, label_dist.values, color='teal')
axes[1, 1].set_title('Number of Labels per Complaint', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Labels')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xticks(label_dist.index)
for i, v in enumerate(label_dist.values):
    axes[1, 1].text(label_dist.index[i], v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'eda_visualizations.png'")

#==============================================================================
# STEP 4: PREPARE LABELS FOR TRAINING
#==============================================================================

print("\n" + "="*80)
print("PREPARING LABELS FOR TRAINING")
print("="*80)

# Define all possible categories (sorted for consistency)
ALL_CATEGORIES = sorted([
    'electrical_issue',
    'internet_issue', 
    'plumbing_issue',
    'furniture_issue',
    'cleanliness_issue',
    'food_issue',
    'noise_issue',
    'billing_issue'
])

print(f"\nAll categories ({len(ALL_CATEGORIES)}):")
for i, cat in enumerate(ALL_CATEGORIES, 1):
    print(f"  {i}. {cat}")

# Convert multi-label to binary matrix (for Issue Classifier)
def labels_to_binary(labels_list, all_categories):
    """Convert list of labels to binary vector"""
    binary = np.zeros(len(all_categories), dtype=int)
    for label in labels_list:
        if label in all_categories:
            idx = all_categories.index(label)
            binary[idx] = 1
    return binary

# Create binary label matrix
y_issues_binary = np.array([labels_to_binary(labels, ALL_CATEGORIES) 
                             for labels in df['labels']])

print(f"\n✓ Binary label matrix created: {y_issues_binary.shape}")
print(f"  (Each row is a binary vector of length {len(ALL_CATEGORIES)})")

# Urgency labels (for Urgency Classifier)
urgency_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_urgency = df['urgency'].map(urgency_mapping).values

print(f"\n✓ Urgency labels encoded: {y_urgency.shape}")
print(f"  Mapping: {urgency_mapping}")

#==============================================================================
# STEP 5: TRAIN-TEST SPLIT
#==============================================================================

print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

# Extract texts
X = df['text'].values

# Split data (80-20 split, stratified by urgency)
X_train, X_test, y_issues_train, y_issues_test, y_urgency_train, y_urgency_test = train_test_split(
    X, 
    y_issues_binary, 
    y_urgency,
    test_size=0.2,
    random_state=42,
    stratify=y_urgency  # Stratify by urgency to maintain distribution
)

print(f"\n✓ Data split completed:")
print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Verify urgency distribution in splits
print(f"\nUrgency distribution in training set:")
train_urgency_counts = pd.Series(y_urgency_train).value_counts(normalize=True).sort_index()
for urgency_val, pct in train_urgency_counts.items():
    urgency_label = [k for k, v in urgency_mapping.items() if v == urgency_val][0]
    print(f"  {urgency_label:8s}: {pct*100:5.1f}%")

print(f"\nUrgency distribution in test set:")
test_urgency_counts = pd.Series(y_urgency_test).value_counts(normalize=True).sort_index()
for urgency_val, pct in test_urgency_counts.items():
    urgency_label = [k for k, v in urgency_mapping.items() if v == urgency_val][0]
    print(f"  {urgency_label:8s}: {pct*100:5.1f}%")

#==============================================================================
# STEP 6: SAVE PROCESSED DATA
#==============================================================================

print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)

# Save as numpy arrays for easy loading
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_issues_train.npy', y_issues_train)
np.save('data/y_issues_test.npy', y_issues_test)
np.save('data/y_urgency_train.npy', y_urgency_train)
np.save('data/y_urgency_test.npy', y_urgency_test)

# Save category mapping
with open('data/category_mapping.json', 'w') as f:
    json.dump({
        'categories': ALL_CATEGORIES,
        'urgency_mapping': urgency_mapping
    }, f, indent=2)

print("\n✓ All processed data saved to 'data/' folder:")
print("  - X_train.npy, X_test.npy")
print("  - y_issues_train.npy, y_issues_test.npy")
print("  - y_urgency_train.npy, y_urgency_test.npy")
print("  - category_mapping.json")

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "="*80)
print("PHASE 1 COMPLETE - SUMMARY")
print("="*80)
print(f"""
Dataset Statistics:
  • Total samples: {len(df)}
  • Training samples: {len(X_train)}
  • Test samples: {len(X_test)}
  • Issue categories: {len(ALL_CATEGORIES)}
  • Urgency levels: 3 (low, medium, high)
  
Multi-label Info:
  • Single-issue: {(df['num_labels']==1).sum()} ({(df['num_labels']==1).sum()/len(df)*100:.1f}%)
  • Multi-issue: {(df['num_labels']>1).sum()} ({(df['num_labels']>1).sum()/len(df)*100:.1f}%)

Next Steps:
  → Phase 2: Feature Engineering (NLP preprocessing + TF-IDF)
  → Phase 3: ML Baseline Models (Logistic Regression + Naive Bayes)
  → Phase 4: Deep Learning Models (LSTM/BERT)
""")

print("="*80)
print("Ready for Phase 2! 🚀")
print("="*80)