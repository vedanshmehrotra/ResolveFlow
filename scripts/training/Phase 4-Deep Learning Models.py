"""
HOSTEL COMPLAINT TRIAGE SYSTEM - PHASE 4: DEEP LEARNING MODELS
================================================================
Author: Vedansh
Date: December 2025

This script trains Deep Learning models:
1. LSTM for Issue Classification (multi-label)
2. LSTM for Urgency Classification (multi-class)
3. Compare with ML baseline from Phase 3

Run this after Phase 3 (ml_baseline.py)
"""

import numpy as np
import json
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, 
    Bidirectional, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Scikit-learn for metrics
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*80)
print("PHASE 4: DEEP LEARNING MODELS (LSTM)")
print("="*80)

#==============================================================================
# STEP 1: LOAD DATA
#==============================================================================

print("\n" + "-"*80)
print("LOADING DATA")
print("-"*80)

try:
    # Load preprocessed text (not TF-IDF, we need raw text for embeddings)
    X_train_processed = np.load('data/X_train_processed.npy', allow_pickle=True)
    X_test_processed = np.load('data/X_test_processed.npy', allow_pickle=True)
    
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
    print(f"  Training samples: {len(X_train_processed)}")
    print(f"  Test samples: {len(X_test_processed)}")
    print(f"  Issue categories: {len(categories)}")
    print(f"  Urgency levels: {len(urgency_mapping)}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required data files not found!")
    print(f"   Missing: {e.filename}")
    print(f"\n   Please run phase2_feature_engineering.py first!")
    exit(1)

#==============================================================================
# STEP 2: TEXT TOKENIZATION & SEQUENCING
#==============================================================================

print("\n" + "-"*80)
print("TOKENIZATION & SEQUENCE PREPARATION")
print("-"*80)

# Hyperparameters
MAX_WORDS = 5000  # Vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Max words per complaint
EMBEDDING_DIM = 128  # Embedding dimension

print(f"\nHyperparameters:")
print(f"  Vocabulary size: {MAX_WORDS}")
print(f"  Max sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")

# Initialize tokenizer
print("\nTokenizing text...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_processed)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_processed)
X_test_seq = tokenizer.texts_to_sequences(X_test_processed)

# Pad sequences to uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Get actual vocabulary size
vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)

print(f"\n✓ Tokenization complete:")
print(f"  Actual vocabulary size: {vocab_size}")
print(f"  Training sequences shape: {X_train_padded.shape}")
print(f"  Test sequences shape: {X_test_padded.shape}")

# Save tokenizer for later use
with open('models/tokenizer_dl.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"  Tokenizer saved to models/tokenizer_dl.pkl")

#==============================================================================
# STEP 3: BUILD ISSUE CLASSIFIER (LSTM - Multi-label)
#==============================================================================

print("\n" + "="*80)
print("MODEL 1: ISSUE CLASSIFIER (LSTM - Multi-label)")
print("="*80)

print("\nBuilding LSTM architecture...")

issue_model = Sequential([
    # Embedding layer
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        name='embedding'
    ),
    
    # Bidirectional LSTM for better context understanding
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='bilstm_1'),
    
    # Global max pooling to capture most important features
    GlobalMaxPooling1D(name='global_max_pool'),
    
    # Dense layers with dropout for regularization
    Dense(64, activation='relu', name='dense_1'),
    Dropout(0.3, name='dropout_1'),
    
    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.2, name='dropout_2'),
    
    # Output layer: 8 neurons for 8 categories (sigmoid for multi-label)
    Dense(len(categories), activation='sigmoid', name='output')
])

# Compile model
issue_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Multi-label classification
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\n✓ Model architecture:")
issue_model.summary()

# Callbacks
callbacks_issue = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        'models/issue_classifier_lstm_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

# Train model
print("\n" + "-"*80)
print("TRAINING ISSUE CLASSIFIER")
print("-"*80)

start_time = time.time()

history_issue = issue_model.fit(
    X_train_padded, y_issues_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=callbacks_issue,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n✓ Training completed in {train_time/60:.2f} minutes")

# Save final model
issue_model.save('models/issue_classifier_lstm.keras')
print("✓ Model saved to models/issue_classifier_lstm.keras")

#==============================================================================
# STEP 4: EVALUATE ISSUE CLASSIFIER
#==============================================================================

print("\n" + "-"*80)
print("EVALUATING ISSUE CLASSIFIER")
print("-"*80)

# Predictions (threshold = 0.5 for multi-label)
y_issues_pred_train = (issue_model.predict(X_train_padded, verbose=0) > 0.5).astype(int)
y_issues_pred_test = (issue_model.predict(X_test_padded, verbose=0) > 0.5).astype(int)
y_issues_proba_test = issue_model.predict(X_test_padded, verbose=0)

# Calculate metrics
train_hamming = hamming_loss(y_issues_train, y_issues_pred_train)
test_hamming = hamming_loss(y_issues_test, y_issues_pred_test)

train_f1_micro = f1_score(y_issues_train, y_issues_pred_train, average='micro')
test_f1_micro = f1_score(y_issues_test, y_issues_pred_test, average='micro')

train_f1_macro = f1_score(y_issues_train, y_issues_pred_train, average='macro')
test_f1_macro = f1_score(y_issues_test, y_issues_pred_test, average='macro')

train_subset_acc = accuracy_score(y_issues_train, y_issues_pred_train)
test_subset_acc = accuracy_score(y_issues_test, y_issues_pred_test)

print("\nOverall Metrics:")
print(f"{'Metric':<25s} {'Training':<15s} {'Test':<15s}")
print("-" * 55)
print(f"{'Hamming Loss':<25s} {train_hamming:<15.4f} {test_hamming:<15.4f}")
print(f"{'Subset Accuracy':<25s} {train_subset_acc:<15.4f} {test_subset_acc:<15.4f}")
print(f"{'F1-Score (Micro)':<25s} {train_f1_micro:<15.4f} {test_f1_micro:<15.4f}")
print(f"{'F1-Score (Macro)':<25s} {train_f1_macro:<15.4f} {test_f1_macro:<15.4f}")

# Per-category metrics
print("\n" + "-"*80)
print("PER-CATEGORY PERFORMANCE (Test Set)")
print("-"*80)

print(f"\n{'Category':<25s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s} {'Support':<10s}")
print("-" * 71)

f1_scores_issue_lstm = []
for i, category in enumerate(categories):
    y_true_cat = y_issues_test[:, i]
    y_pred_cat = y_issues_pred_test[:, i]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, average=None, zero_division=0
    )
    
    if len(precision) > 1:
        precision_pos = precision[1]
        recall_pos = recall[1]
        f1_pos = f1[1]
        support_pos = support[1]
    else:
        precision_pos = precision[0]
        recall_pos = recall[0]
        f1_pos = f1[0]
        support_pos = support[0]
    
    f1_scores_issue_lstm.append(f1_pos)
    print(f"{category:<25s} {precision_pos:<12.3f} {recall_pos:<12.3f} {f1_pos:<12.3f} {int(support_pos):<10d}")

#==============================================================================
# STEP 5: BUILD URGENCY CLASSIFIER (LSTM - Multi-class)
#==============================================================================

print("\n" + "="*80)
print("MODEL 2: URGENCY CLASSIFIER (LSTM - Multi-class)")
print("="*80)

print("\nBuilding LSTM architecture...")

urgency_model = Sequential([
    # Embedding layer (can reuse same architecture)
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        name='embedding'
    ),
    
    # Bidirectional LSTM
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='bilstm_1'),
    
    # Global max pooling
    GlobalMaxPooling1D(name='global_max_pool'),
    
    # Dense layers
    Dense(64, activation='relu', name='dense_1'),
    Dropout(0.3, name='dropout_1'),
    
    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.2, name='dropout_2'),
    
    # Output layer: 3 neurons for 3 urgency levels (softmax for multi-class)
    Dense(3, activation='softmax', name='output')
])

# Compile model
urgency_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)

print("\n✓ Model architecture:")
urgency_model.summary()

# Callbacks
callbacks_urgency = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        'models/urgency_classifier_lstm_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

# Train model
print("\n" + "-"*80)
print("TRAINING URGENCY CLASSIFIER")
print("-"*80)

start_time = time.time()

history_urgency = urgency_model.fit(
    X_train_padded, y_urgency_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=callbacks_urgency,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n✓ Training completed in {train_time/60:.2f} minutes")

# Save final model
urgency_model.save('models/urgency_classifier_lstm.keras')
print("✓ Model saved to models/urgency_classifier_lstm.keras")

#==============================================================================
# STEP 6: EVALUATE URGENCY CLASSIFIER
#==============================================================================

print("\n" + "-"*80)
print("EVALUATING URGENCY CLASSIFIER")
print("-"*80)

# Predictions
y_urgency_pred_train = np.argmax(urgency_model.predict(X_train_padded, verbose=0), axis=1)
y_urgency_pred_test = np.argmax(urgency_model.predict(X_test_padded, verbose=0), axis=1)
y_urgency_proba_test = urgency_model.predict(X_test_padded, verbose=0)

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
urgency_labels = ['low', 'medium', 'high']
print("\n" + "-"*80)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("-"*80)
print("\n" + classification_report(
    y_urgency_test, 
    y_urgency_pred_test,
    target_names=urgency_labels,
    digits=3
))

# Confusion matrix
cm_urgency = confusion_matrix(y_urgency_test, y_urgency_pred_test)

#==============================================================================
# STEP 7: COMPARE WITH ML BASELINE
#==============================================================================

print("\n" + "="*80)
print("COMPARISON: LSTM vs ML BASELINE")
print("="*80)

try:
    # Load ML baseline metrics
    with open('results/ml_baseline_metrics.json', 'r') as f:
        ml_metrics = json.load(f)
    
    print("\n" + "-"*80)
    print("ISSUE CLASSIFIER COMPARISON")
    print("-"*80)
    
    ml_issue_f1 = ml_metrics['issue_classifier']['test_f1_macro']
    lstm_issue_f1 = test_f1_macro
    improvement = ((lstm_issue_f1 - ml_issue_f1) / ml_issue_f1) * 100
    
    print(f"\n{'Model':<20s} {'F1-Macro':<15s} {'Subset Acc':<15s} {'Hamming Loss':<15s}")
    print("-" * 65)
    print(f"{'ML Baseline (LR)':<20s} {ml_issue_f1:<15.4f} {ml_metrics['issue_classifier']['test_subset_accuracy']:<15.4f} {ml_metrics['issue_classifier']['test_hamming_loss']:<15.4f}")
    print(f"{'LSTM':<20s} {lstm_issue_f1:<15.4f} {test_subset_acc:<15.4f} {test_hamming:<15.4f}")
    print(f"\n{'Improvement:':<20s} {improvement:+.2f}%")
    
    print("\n" + "-"*80)
    print("URGENCY CLASSIFIER COMPARISON")
    print("-"*80)
    
    ml_urgency_acc = ml_metrics['urgency_classifier']['test_accuracy']
    lstm_urgency_acc = test_acc
    improvement = ((lstm_urgency_acc - ml_urgency_acc) / ml_urgency_acc) * 100
    
    print(f"\n{'Model':<20s} {'Accuracy':<15s} {'F1-Weighted':<15s}")
    print("-" * 50)
    print(f"{'ML Baseline (NB)':<20s} {ml_urgency_acc:<15.4f} {ml_metrics['urgency_classifier']['test_f1_weighted']:<15.4f}")
    print(f"{'LSTM':<20s} {lstm_urgency_acc:<15.4f} {test_f1:<15.4f}")
    print(f"\n{'Improvement:':<20s} {improvement:+.2f}%")
    
except FileNotFoundError:
    print("\n⚠️  ML baseline metrics not found. Skipping comparison.")
    ml_metrics = None

#==============================================================================
# STEP 8: VISUALIZATIONS
#==============================================================================

print("\n" + "-"*80)
print("GENERATING VISUALIZATIONS")
print("-"*80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Training history - Issue Classifier Loss
ax1 = plt.subplot(3, 3, 1)
ax1.plot(history_issue.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history_issue.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Issue Classifier: Training History (Loss)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training history - Issue Classifier Accuracy
ax2 = plt.subplot(3, 3, 2)
ax2.plot(history_issue.history['accuracy'], label='Train Accuracy', linewidth=2)
ax2.plot(history_issue.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Issue Classifier: Training History (Accuracy)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Training history - Urgency Classifier Loss
ax3 = plt.subplot(3, 3, 3)
ax3.plot(history_urgency.history['loss'], label='Train Loss', linewidth=2)
ax3.plot(history_urgency.history['val_loss'], label='Val Loss', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Urgency Classifier: Training History (Loss)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Training history - Urgency Classifier Accuracy
ax4 = plt.subplot(3, 3, 4)
ax4.plot(history_urgency.history['accuracy'], label='Train Accuracy', linewidth=2)
ax4.plot(history_urgency.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy')
ax4.set_title('Urgency Classifier: Training History (Accuracy)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Issue Classifier - Per-category F1 comparison
ax5 = plt.subplot(3, 3, 5)
if ml_metrics:
    ml_f1_per_cat = [ml_metrics['issue_classifier']['per_category_f1'][cat] for cat in categories]
    x = np.arange(len(categories))
    width = 0.35
    ax5.barh(x - width/2, ml_f1_per_cat, width, label='ML Baseline', alpha=0.8)
    ax5.barh(x + width/2, f1_scores_issue_lstm, width, label='LSTM', alpha=0.8)
    ax5.set_yticks(x)
    ax5.set_yticklabels(categories)
else:
    ax5.barh(categories, f1_scores_issue_lstm, color='steelblue')
ax5.set_xlabel('F1-Score')
ax5.set_title('Issue Classifier: F1-Score per Category', fontweight='bold')
ax5.legend()
ax5.set_xlim([0, 1])

# Plot 6: Urgency Confusion Matrix
ax6 = plt.subplot(3, 3, 6)
sns.heatmap(cm_urgency, annot=True, fmt='d', cmap='Blues',
            xticklabels=urgency_labels, yticklabels=urgency_labels,
            ax=ax6, cbar_kws={'label': 'Count'})
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')
ax6.set_title('Urgency Classifier: Confusion Matrix', fontweight='bold')

# Plot 7: Model Comparison - Issue Classifier
ax7 = plt.subplot(3, 3, 7)
if ml_metrics:
    models = ['ML\nBaseline', 'LSTM']
    f1_scores = [ml_metrics['issue_classifier']['test_f1_macro'], test_f1_macro]
    colors = ['lightcoral', 'lightgreen']
    bars = ax7.bar(models, f1_scores, color=colors, alpha=0.7)
    ax7.set_ylabel('F1-Score (Macro)')
    ax7.set_title('Issue Classifier: Model Comparison', fontweight='bold')
    ax7.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 8: Model Comparison - Urgency Classifier
ax8 = plt.subplot(3, 3, 8)
if ml_metrics:
    models = ['ML\nBaseline', 'LSTM']
    accuracies = [ml_metrics['urgency_classifier']['test_accuracy'], test_acc]
    colors = ['lightcoral', 'lightgreen']
    bars = ax8.bar(models, accuracies, color=colors, alpha=0.7)
    ax8.set_ylabel('Accuracy')
    ax8.set_title('Urgency Classifier: Model Comparison', fontweight='bold')
    ax8.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 9: Prediction confidence distribution - Urgency
ax9 = plt.subplot(3, 3, 9)
correct_mask = (y_urgency_pred_test == y_urgency_test)
correct_confidences = y_urgency_proba_test[correct_mask].max(axis=1)
incorrect_confidences = y_urgency_proba_test[~correct_mask].max(axis=1)
ax9.hist([correct_confidences, incorrect_confidences],
         bins=20, label=['Correct', 'Incorrect'],
         color=['green', 'red'], alpha=0.6)
ax9.set_xlabel('Prediction Confidence')
ax9.set_ylabel('Count')
ax9.set_title('Urgency: Prediction Confidence (LSTM)', fontweight='bold')
ax9.legend()

plt.tight_layout()
plt.savefig('results/dl_lstm_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'results/dl_lstm_results.png'")

#==============================================================================
# STEP 9: SAVE RESULTS
#==============================================================================

print("\n" + "-"*80)
print("SAVING RESULTS")
print("-"*80)

# Save LSTM metrics
lstm_results = {
    'issue_classifier': {
        'train_hamming_loss': float(train_hamming),
        'test_hamming_loss': float(test_hamming),
        'train_f1_micro': float(train_f1_micro),
        'test_f1_micro': float(test_f1_micro),
        'train_f1_macro': float(train_f1_macro),
        'test_f1_macro': float(test_f1_macro),
        'train_subset_accuracy': float(train_subset_acc),
        'test_subset_accuracy': float(test_subset_acc),
        'per_category_f1': {cat: float(score) for cat, score in zip(categories, f1_scores_issue_lstm)}
    },
    'urgency_classifier': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_f1_weighted': float(train_f1),
        'test_f1_weighted': float(test_f1)
    },
    'training_history': {
        'issue_loss': [float(x) for x in history_issue.history['loss']],
        'issue_val_loss': [float(x) for x in history_issue.history['val_loss']],
        'urgency_loss': [float(x) for x in history_urgency.history['loss']],
        'urgency_val_loss': [float(x) for x in history_urgency.history['val_loss']]
    }
}

with open('results/dl_lstm_metrics.json', 'w') as f:
    json.dump(lstm_results, f, indent=2)

print("\n✓ Results saved:")
print("  - models/issue_classifier_lstm.keras")
print("  - models/urgency_classifier_lstm.keras")
print("  - models/tokenizer_dl.pkl")
print("  - results/dl_lstm_metrics.json")
print("  - results/dl_lstm_results.png")

#==============================================================================
# STEP 10: EXAMPLE PREDICTIONS
#==============================================================================

print("\n" + "-"*80)
print("EXAMPLE PREDICTIONS")
print("-"*80)

# Load original text
X_test_original = np.load('data/X_test.npy', allow_pickle=True)

print("\nShowing 3 random test examples:\n")

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
    print(f"\nPredicted (LSTM):")
    print(f"  Issues:  {', '.join(pred_issues)}")
    print(f"  Urgency: {pred_urgency} (confidence: {pred_urgency_conf:.2f})")
    print()

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "="*80)
print("PHASE 4 COMPLETE - DEEP LEARNING MODELS SUMMARY")
print("="*80)

if ml_metrics:
    ml_issue_f1 = ml_metrics