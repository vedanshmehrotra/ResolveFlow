import ollama
import random
import json

# -----------------------------
# ISSUE DEFINITIONS
# -----------------------------
issue_scenarios = {
    "electrical_issue": [
        "the fan in my room has stopped working",
        "the AC is not cooling properly",
        "the light switch is broken",
        "there's frequent power tripping in my room",
        "the room heater is not working"
    ],
    "internet_issue": [
        "the hostel wifi is not connecting",
        "the internet speed is extremely slow",
        "my room has no wifi signal",
        "the router in my corridor is not working"
    ],
    "plumbing_issue": [
        "there is water leakage from the bathroom ceiling",
        "the bathroom tap won't stop dripping",
        "the toilet is clogged",
        "there's no hot water supply",
        "the sink drain is blocked"
    ],
    "furniture_issue": [
        "the bed frame is broken",
        "my chair has a broken leg",
        "the cupboard door won't close",
        "the study table is wobbly",
        "the window latch is damaged"
    ],
    "cleanliness_issue": [
        "the washroom has not been cleaned for days",
        "there's garbage piling up in the corridor",
        "the common room floor is very dirty",
        "there are cockroaches in my room"
    ],
    "food_issue": [
        "the mess food quality is very poor",
        "the food served was stale",
        "there's insufficient food in the mess",
        "the food is always cold"
    ],
    "noise_issue": [
        "the room next door is extremely noisy at night",
        "there's construction noise starting very early",
        "people are playing loud music in the corridor"
    ],
    "billing_issue": [
        "there's an error in my hostel fee bill",
        "I was charged twice for mess fees",
        "the electricity bill amount seems incorrect"
    ]
}

# -----------------------------
# LINGUISTIC URGENCY DEFINITIONS
# -----------------------------
# These define HOW the student writes, not objective severity

URGENCY_CONTEXTS = {
    "low": {
        "phrases": [
            "whenever you get time",
            "no immediate rush",
            "please look into this when possible",
            "not urgent but",
            "at your convenience"
        ],
        "contexts": [
            "it's a minor inconvenience",
            "it's not a big deal but",
            "just wanted to inform you",
            "this can wait but",
            "no emergency but"
        ]
    },
    "medium": {
        "phrases": [
            "please fix this soon",
            "this needs attention",
            "hoping this gets resolved this week",
            "please address this",
            "this should be fixed"
        ],
        "contexts": [
            "it's been like this for a few days",
            "it's getting annoying",
            "I have work to do and",
            "this is causing problems",
            "it's quite uncomfortable"
        ]
    },
    "high": {
        "phrases": [
            "this is urgent",
            "please fix this immediately",
            "I need this resolved ASAP",
            "this is an emergency",
            "need this fixed before tomorrow"
        ],
        "contexts": [
            "I have an exam tomorrow and",
            "I have an assignment due and",
            "I can't sleep because of this",
            "this is completely unusable",
            "this is a serious problem"
        ]
    }
}

# -----------------------------
# MODEL CALL
# -----------------------------
def call_model(prompt):
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"].strip()

# -----------------------------
# VALIDATION
# -----------------------------
def validate_entry(text, labels, urgency):
    """Check if urgency language matches assigned urgency"""
    text_lower = text.lower()
    
    # High urgency indicators
    high_words = ["urgent", "immediately", "asap", "emergency", "critical", "exam", "assignment due"]
    # Low urgency indicators  
    low_words = ["whenever", "no rush", "not urgent", "minor", "convenience", "no emergency"]
    
    high_count = sum(1 for word in high_words if word in text_lower)
    low_count = sum(1 for word in low_words if word in text_lower)
    
    # Validation logic
    if urgency == "high" and (low_count > 0 or high_count == 0):
        return False
    if urgency == "low" and (high_count > 0 or low_count == 0):
        return False
    if urgency == "medium" and (high_count > 1 or low_count > 0):
        return False
        
    # Check minimum length
    if len(text.split()) < 20:
        return False
        
    return True

# -----------------------------
# GENERATION
# -----------------------------
def generate_complaint(labels, urgency):
    """Generate complaint with specific linguistic urgency"""
    
    # Get issue descriptions
    issues = [random.choice(issue_scenarios[label]) for label in labels]
    issues_text = " and ".join(issues)
    
    # Get urgency language elements
    phrase = random.choice(URGENCY_CONTEXTS[urgency]["phrases"])
    context = random.choice(URGENCY_CONTEXTS[urgency]["contexts"])
    
    # Build detailed prompt
    prompt = f"""Write a hostel complaint email body from a college student about:
{issues_text}

The student feels this is {urgency} priority. Show this by:
- Including this phrase: "{phrase}"
- Adding context like: "{context}"

Style requirements:
- Write 4-5 sentences
- Use simple, natural student language
- Describe the problem and its impact
- Make the urgency level clear through tone and word choice
- NO greetings, signatures, or subject lines
- Just the complaint body

Example for HIGH urgency:
"The wifi in my room has completely stopped working and I have an assignment submission tomorrow. I've tried everything but can't connect at all. This is urgent because I need to upload my project files before the deadline. Please fix this immediately as I'm getting really stressed about missing my submission."

Example for LOW urgency:
"The chair in my room has a wobbly leg but it's still usable. It's not a big problem but I thought I should report it. Whenever you get time, it would be good to have it fixed. No immediate rush though, I can manage for now."

Now write the complaint:"""

    text = call_model(prompt)
    
    # Clean up
    text = text.replace("Subject:", "").replace("Dear", "").strip()
    # Remove any lines that look like headers
    lines = [l for l in text.split('\n') if l.strip() and not l.strip().endswith(':')]
    text = ' '.join(lines)
    
    return text

# -----------------------------
# MAIN GENERATION
# -----------------------------
TOTAL = 200
dataset = []
attempts = 0
max_attempts = TOTAL * 4

print("Generating hostel complaint dataset...")
print("Note: Urgency labels represent LINGUISTIC urgency (how the student writes)")
print("      NOT objective severity\n")

# Distribution targets
urgency_targets = {"low": 0.3, "medium": 0.5, "high": 0.2}
urgency_counts = {"low": 0, "medium": 0, "high": 0}

while len(dataset) < TOTAL and attempts < max_attempts:
    attempts += 1
    
    # Select issue labels
    all_labels = list(issue_scenarios.keys())
    if random.random() < 0.7:
        labels = [random.choice(all_labels)]
    else:
        labels = random.sample(all_labels, 2)
    
    # Select urgency to maintain distribution
    current_dist = {k: v/len(dataset) if len(dataset) > 0 else 0 
                   for k, v in urgency_counts.items()}
    
    # Pick urgency that we need more of
    needed = [(u, urgency_targets[u] - current_dist[u]) for u in ["low", "medium", "high"]]
    needed.sort(key=lambda x: x[1], reverse=True)
    urgency = needed[0][0]
    
    # Generate
    try:
        text = generate_complaint(labels, urgency)
        
        if validate_entry(text, labels, urgency):
            dataset.append({
                "text": text,
                "labels": labels,
                "urgency": urgency
            })
            urgency_counts[urgency] += 1
            print(f"✓ Generated {len(dataset)}/{TOTAL} | {urgency:6} | {', '.join(labels)}")
        else:
            print(f"✗ Validation failed (attempt {attempts})")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        continue

# -----------------------------
# SAVE
# -----------------------------
output_file = "hostel_dataset_linguistic_urgency_batch3.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"COMPLETE: {len(dataset)} entries saved to {output_file}")
print(f"Success rate: {len(dataset)/attempts*100:.1f}%")
print(f"{'='*60}")

# Statistics
print("\nDataset Statistics:")
print("-" * 60)

label_counts = {}
for entry in dataset:
    for label in entry["labels"]:
        label_counts[label] = label_counts.get(label, 0) + 1

print("\nIssue Type Distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:20} : {count:3} ({count/len(dataset)*100:.1f}%)")

print("\nLinguistic Urgency Distribution:")
for urgency in ["low", "medium", "high"]:
    count = urgency_counts[urgency]
    print(f"  {urgency:6} : {count:3} ({count/len(dataset)*100:.1f}%)")

print("\n" + "="*60)
print("IMPORTANT NOTE:")
print("Urgency labels represent HOW URGENTLY the student wrote,")
print("not objective severity. Final urgency should be assigned by")
print("human operators considering: deadlines, severity, resources.")
print("="*60)

import json
import re

# Load dataset
with open('merged-1766558695221.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original dataset: {len(data)} rows")

# -----------------------------
# VALIDATION & CORRECTION RULES
# -----------------------------

def validate_and_fix(entry):
    """
    Validate entry and return (is_valid, fixed_entry, reason)
    """
    text = entry["text"].lower()
    labels = entry["labels"]
    urgency = entry["urgency"]
    
    # Rule 1: Fix "power tripping" social issues
    if "electrical_issue" in labels and ("dominating" in text or "interrupting" in text or 
                                         "opinions" in text or "respect" in text):
        # This is NOT electrical - it's misclassified
        return False, entry, "power_tripping_social_issue"
    
    # Rule 2: Fix construction noise labeled as electrical
    if "electrical_issue" in labels and ("construction noise" in text or "drilling" in text or 
                                         "hammering" in text):
        entry["labels"] = ["noise_issue"]
        return True, entry, "fixed_construction_noise"
    
    # Rule 3: Fix music/noise labeled as electrical
    if "electrical_issue" in labels and ("loud music" in text or "blasting music" in text or 
                                         "noise" in text and "power" not in text):
        entry["labels"] = ["noise_issue"]
        return True, entry, "fixed_music_noise"
    
    # Rule 4: Remove furniture_issue if AC/fan/heater mentioned but no furniture
    if "furniture_issue" in labels and "electrical_issue" in labels:
        furniture_words = ["chair", "table", "bed", "cupboard", "window", "latch", "door"]
        has_furniture = any(word in text for word in furniture_words)
        
        if not has_furniture:
            entry["labels"] = ["electrical_issue"]
            return True, entry, "removed_spurious_furniture_label"
    
    # Rule 5: Check urgency-text mismatch
    high_words = ["urgent", "immediately", "asap", "emergency", "exam tomorrow", "before tomorrow"]
    low_words = ["whenever", "no rush", "not urgent", "convenience", "minor"]
    
    has_high = any(word in text for word in high_words)
    has_low = any(word in text for word in low_words)
    
    if urgency == "high" and has_low and not has_high:
        # Urgency contradiction
        return False, entry, "urgency_contradiction"
    
    if urgency == "low" and has_high and not has_low:
        # Urgency contradiction
        return False, entry, "urgency_contradiction"
    
    # Rule 6: Unrealistic high urgency for minor issues
    minor_issues = ["cupboard door", "chair leg", "window latch", "study table wobbly"]
    if urgency == "high" and any(issue in text for issue in minor_issues):
        # Downgrade to medium
        entry["urgency"] = "medium"
        return True, entry, "downgraded_unrealistic_urgency"
    
    return True, entry, "valid"

# -----------------------------
# PROCESS DATASET
# -----------------------------

cleaned = []
removed = []
fixes = {
    "valid": 0,
    "fixed_construction_noise": 0,
    "fixed_music_noise": 0,
    "removed_spurious_furniture_label": 0,
    "downgraded_unrealistic_urgency": 0,
    "power_tripping_social_issue": 0,
    "urgency_contradiction": 0
}

for entry in data:
    is_valid, fixed_entry, reason = validate_and_fix(entry)
    
    fixes[reason] += 1
    
    if is_valid:
        cleaned.append(fixed_entry)
    else:
        removed.append({"entry": entry, "reason": reason})

# -----------------------------
# SAVE RESULTS
# -----------------------------

output_file = "hostel_dataset_validated.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

# Save removed entries for review
with open("removed_entries.json", 'w', encoding='utf-8') as f:
    json.dump(removed, f, indent=2, ensure_ascii=False)

# -----------------------------
# REPORT
# -----------------------------

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
print(f"Original:    {len(data)}")
print(f"Cleaned:     {len(cleaned)}")
print(f"Removed:     {len(removed)}")
print(f"Retention:   {len(cleaned)/len(data)*100:.1f}%")

print("\n" + "="*60)
print("FIXES APPLIED:")
print("="*60)
for reason, count in sorted(fixes.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {reason:40} : {count:4}")

# Statistics
print("\n" + "="*60)
print("CLEANED DATASET STATS:")
print("="*60)

label_counts = {}
for entry in cleaned:
    for label in entry["labels"]:
        label_counts[label] = label_counts.get(label, 0) + 1

urgency_counts = {"low": 0, "medium": 0, "high": 0}
for entry in cleaned:
    urgency_counts[entry["urgency"]] += 1

print("\nIssue Distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:20} : {count:4} ({count/len(cleaned)*100:.1f}%)")

print("\nUrgency Distribution:")
for urgency, count in urgency_counts.items():
    print(f"  {urgency:8} : {count:4} ({count/len(cleaned)*100:.1f}%)")

print("\n" + "="*60)
print(f"✅ Saved to: {output_file}")
print(f"📋 Removed entries saved to: removed_entries.json")
print("="*60)

# Show sample removed entries
if removed:
    print("\n" + "="*60)
    print("SAMPLE REMOVED ENTRIES (First 5):")
    print("="*60)
    for i, item in enumerate(removed[:5], 1):
        print(f"\n{i}. Reason: {item['reason']}")
        print(f"   Text: {item['entry']['text'][:100]}...")
        print(f"   Labels: {item['entry']['labels']}")
        print(f"   Urgency: {item['entry']['urgency']}")

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
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_issues_train.npy', y_issues_train)
np.save('y_issues_test.npy', y_issues_test)
np.save('y_urgency_train.npy', y_urgency_train)
np.save('y_urgency_test.npy', y_urgency_test)

# Save category mapping
with open('category_mapping.json', 'w') as f:
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

"""
HOSTEL COMPLAINT TRIAGE SYSTEM - PHASE 2: FEATURE ENGINEERING
==============================================================
Author: Your Name
Date: December 2024

This script handles:
1. Text preprocessing (cleaning, tokenization, lemmatization)
2. TF-IDF vectorization
3. Feature extraction for ML models
"""

import numpy as np
import pandas as pd
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("\n" + "="*80)
print("PHASE 2: NLP FEATURE ENGINEERING")
print("="*80)

#==============================================================================
# STEP 1: LOAD PREPROCESSED DATA
#==============================================================================

print("\n" + "-"*80)
print("LOADING PREPROCESSED DATA")
print("-"*80)

X_train = np.load('data/X_train.npy', allow_pickle=True)
X_test = np.load('data/X_test.npy', allow_pickle=True)

print(f"\n✓ Data loaded:")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

#==============================================================================
# STEP 2: TEXT PREPROCESSING
#==============================================================================

print("\n" + "-"*80)
print("TEXT PREPROCESSING")
print("-"*80)

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep some important words that are often stopwords but matter for complaints
important_words = {'not', 'no', 'need', 'very', 'too', 'much', 'more', 'most'}
stop_words = stop_words - important_words

def preprocess_text(text):
    """
    Comprehensive text preprocessing pipeline:
    1. Lowercase
    2. Remove special characters (keep basic punctuation)
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters but keep apostrophes and basic punctuation
    text = re.sub(r'[^a-zA-Z\s\'\-]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    # Rejoin
    return ' '.join(tokens)

print("\nPreprocessing training data...")
X_train_processed = [preprocess_text(text) for text in X_train]

print("Preprocessing test data...")
X_test_processed = [preprocess_text(text) for text in X_test]

print("\n✓ Text preprocessing complete")

# Show example
print("\nExample preprocessing:")
print(f"\nOriginal:\n{X_train[0][:200]}...")
print(f"\nProcessed:\n{X_train_processed[0][:200]}...")

#==============================================================================
# STEP 3: TF-IDF VECTORIZATION
#==============================================================================

print("\n" + "-"*80)
print("TF-IDF VECTORIZATION")
print("-"*80)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,      # Limit to top 2000 features
    ngram_range=(1, 2),     # Use unigrams and bigrams
    min_df=3,               # Ignore terms that appear in fewer than 3 documents
    max_df=0.8,             # Ignore terms that appear in more than 80% of documents
    sublinear_tf=True       # Apply sublinear tf scaling (1 + log(tf))
)

print("\nFitting TF-IDF vectorizer on training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed)

print("Transforming test data...")
X_test_tfidf = tfidf_vectorizer.transform(X_test_processed)

print(f"\n✓ TF-IDF vectorization complete:")
print(f"  Training shape: {X_train_tfidf.shape}")
print(f"  Test shape: {X_test_tfidf.shape}")
print(f"  Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"  Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.2f}%")

# Display top features
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\nSample features (first 20):")
print(", ".join(feature_names[:20]))

# Get most important features (highest average TF-IDF scores)
mean_tfidf = X_train_tfidf.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-20:][::-1]
print(f"\nTop 20 most important features:")
for idx in top_indices:
    print(f"  {feature_names[idx]:20s}: {mean_tfidf[idx]:.4f}")

#==============================================================================
# STEP 4: FEATURE ANALYSIS
#==============================================================================

print("\n" + "-"*80)
print("FEATURE ANALYSIS")
print("-"*80)

# Analyze features per category
y_issues_train = np.load('data/y_issues_train.npy')
with open('data/category_mapping.json', 'r') as f:
    mappings = json.load(f)
    categories = mappings['categories']

print("\nMost important features per issue category:")
print("(Top 10 features with highest average TF-IDF per category)\n")

for i, category in enumerate(categories):
    # Get complaints with this category
    mask = y_issues_train[:, i] == 1
    if mask.sum() > 0:
        # Calculate mean TF-IDF for this category
        category_tfidf = X_train_tfidf[mask].mean(axis=0).A1
        top_idx = category_tfidf.argsort()[-10:][::-1]
        
        print(f"{category}:")
        features = [f"{feature_names[idx]} ({category_tfidf[idx]:.3f})" 
                   for idx in top_idx]
        print(f"  {', '.join(features)}\n")

#==============================================================================
# STEP 5: SAVE FEATURES AND VECTORIZER
#==============================================================================

print("\n" + "-"*80)
print("SAVING FEATURES AND VECTORIZER")
print("-"*80)

# Save TF-IDF features
from scipy import sparse
sparse.save_npz('X_train_tfidf.npz', X_train_tfidf)
sparse.save_npz('X_test_tfidf.npz', X_test_tfidf)

# Save processed text (for deep learning later)
np.save('X_train_processed.npy', X_train_processed)
np.save('X_test_processed.npy', X_test_processed)

# Save vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("\n✓ All features and vectorizer saved:")
print("  - X_train_tfidf.npz, X_test_tfidf.npz")
print("  - X_train_processed.npy, X_test_processed.npy")
print("  - tfidf_vectorizer.pkl")

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "="*80)
print("PHASE 2 COMPLETE - SUMMARY")
print("="*80)
print(f"""
Feature Engineering Results:
  • Preprocessed texts: {len(X_train_processed)} train, {len(X_test_processed)} test
  • TF-IDF features: {X_train_tfidf.shape[1]}
  • N-gram range: unigrams + bigrams
  • Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}
  
Feature Characteristics:
  • Max features: 2000
  • Min document frequency: 3
  • Max document frequency: 80%
  • Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.1f}%

Next Steps:
  → Phase 3: Train ML Baseline Models
     - Issue Classifier: Logistic Regression (multi-label)
     - Urgency Classifier: Naive Bayes (multi-class)
""")

print("="*80)
print("Ready for Phase 3! 🚀")
print("="*80)