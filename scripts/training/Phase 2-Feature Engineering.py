"""
HOSTEL COMPLAINT TRIAGE SYSTEM - PHASE 2: FEATURE ENGINEERING
==============================================================
Author: Vedansh
Date: December 2025

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
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # For lemmatization
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
sparse.save_npz('data/X_train_tfidf.npz', X_train_tfidf)
sparse.save_npz('data/X_test_tfidf.npz', X_test_tfidf)

# Save processed text (for deep learning later)
np.save('data/X_train_processed.npy', X_train_processed)
np.save('data/X_test_processed.npy', X_test_processed)

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