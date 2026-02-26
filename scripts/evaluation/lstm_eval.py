"""
LSTM Evaluation on 120 test samples + LR 5-fold CV comparison
"""
import os, pickle, json, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy import sparse
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 100
cats = None
ul   = ['low', 'medium', 'high']

# ── Load data ────────────────────────────────────────────────────────
X_test_processed = np.load('data/X_test_processed.npy', allow_pickle=True)
y_issues_test    = np.load('data/y_issues_test.npy')        # (120, 8)
y_urgency_test   = np.load('data/y_urgency_test.npy')       # (120,)
X_train_tfidf    = sparse.load_npz('data/X_train_tfidf.npz')
X_test_tfidf     = sparse.load_npz('data/X_test_tfidf.npz')
y_issues_train   = np.load('data/y_issues_train.npy')
y_urgency_train  = np.load('data/y_urgency_train.npy')

with open('data/category_mapping.json') as f:
    m = json.load(f)
cats = m['categories']

# ── Load models ───────────────────────────────────────────────────────
with open('models/tokenizer_dl.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
issue_lstm   = load_model('models/issue_classifier_lstm_best.keras',   compile=False)
urgency_lstm = load_model('models/urgency_classifier_lstm_best.keras', compile=False)
with open('models/issue_classifier_ml.pkl', 'rb') as f:
    lr_clf = pickle.load(f)
with open('models/urgency_classifier_ml.pkl', 'rb') as f:
    nb_clf = pickle.load(f)

# ── Tokenise test set ─────────────────────────────────────────────────
seqs   = tokenizer.texts_to_sequences(X_test_processed)
padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# ── LSTM Issue predictions ────────────────────────────────────────────
issue_proba_lstm = issue_lstm.predict(padded, verbose=0)          # (120, 8)
issue_pred_lstm  = (issue_proba_lstm >= 0.5).astype(int)

# ── LSTM Urgency predictions ──────────────────────────────────────────
urgency_proba_lstm = urgency_lstm.predict(padded, verbose=0)      # (120, 3)
urgency_pred_lstm  = urgency_proba_lstm.argmax(axis=1)            # (120,)

out = []

# ════════════════════════════════════════════════════════════════
# 1. LSTM ISSUE METRICS
# ════════════════════════════════════════════════════════════════
hl   = hamming_loss(y_issues_test, issue_pred_lstm)
sa   = accuracy_score(y_issues_test, issue_pred_lstm)
f1mi = f1_score(y_issues_test, issue_pred_lstm, average='micro', zero_division=0)
f1ma = f1_score(y_issues_test, issue_pred_lstm, average='macro', zero_division=0)

out.append("=== LSTM ISSUE CLASSIFIER (120 test samples) ===")
out.append(f"Hamming Loss    : {hl:.6f}")
out.append(f"Subset Accuracy : {sa:.4f}")
out.append(f"Micro F1        : {f1mi:.4f}")
out.append(f"Macro F1        : {f1ma:.4f}")

# ── Per-class ─────────────────────────────────────────────────────────
out.append("\nPER-CLASS  (t=0.5)")
out.append(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
out.append("-" * 65)
for i, cat in enumerate(cats):
    p, r, f, s = precision_recall_fscore_support(
        y_issues_test[:, i], issue_pred_lstm[:, i],
        average=None, zero_division=0
    )
    if len(p) > 1:
        out.append(f"{cat:<25} {p[1]:>10.4f} {r[1]:>10.4f} {f[1]:>10.4f} {int(s[1]):>10d}")
    else:
        out.append(f"{cat:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {int(s[0]):>10d}")

# ════════════════════════════════════════════════════════════════
# 2. LSTM URGENCY METRICS
# ════════════════════════════════════════════════════════════════
urg_acc = accuracy_score(y_urgency_test, urgency_pred_lstm)
urg_f1w = f1_score(y_urgency_test, urgency_pred_lstm, average='weighted', zero_division=0)
urg_f1m = f1_score(y_urgency_test, urgency_pred_lstm, average='macro',    zero_division=0)
cm_lstm = confusion_matrix(y_urgency_test, urgency_pred_lstm)

out.append("\n=== LSTM URGENCY CLASSIFIER (120 test samples) ===")
out.append(f"Accuracy    : {urg_acc:.4f}")
out.append(f"Macro F1    : {urg_f1m:.4f}")
out.append(f"Weighted F1 : {urg_f1w:.4f}")
out.append("\nConfusion Matrix:")
out.append(f"{'':12} {'Pred_low':>10} {'Pred_med':>10} {'Pred_high':>10}")
for i, lbl in enumerate(ul):
    out.append(f"{'True_'+lbl:12} {cm_lstm[i,0]:>10d} {cm_lstm[i,1]:>10d} {cm_lstm[i,2]:>10d}")

# Per-class P/R/F1 for urgency LSTM
out.append("\nPer-class (Urgency LSTM):")
out.append(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
out.append("-" * 45)
p_u, r_u, f_u, s_u = precision_recall_fscore_support(
    y_urgency_test, urgency_pred_lstm, average=None, zero_division=0
)
for i, lbl in enumerate(ul):
    out.append(f"{lbl:<12} {p_u[i]:>10.4f} {r_u[i]:>10.4f} {f_u[i]:>10.4f}")

# ════════════════════════════════════════════════════════════════
# 3. LR 5-FOLD CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════
out.append("\n=== 5-FOLD CV: Logistic Regression (issue, macro F1) ===")
out.append("Note: CV uses full 597-sample dataset to get unbiased estimate.")

# For multi-label CV we need to do it manually (sklearn CV doesn't natively
# support multi-label macro F1 directly via cross_val_score)
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Load full TF-IDF features (train + test combined)
X_all = sparse.vstack([X_train_tfidf, X_test_tfidf])
y_all = np.vstack([y_issues_train, y_issues_test])
y_urg_all = np.concatenate([y_urgency_train, y_urgency_test])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_macro = []
cv_f1_micro = []
cv_hamming   = []

fold = 0
for tr_idx, te_idx in kf.split(X_all):
    fold += 1
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                           random_state=42, solver='lbfgs'),
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    f1ma = f1_score(y_te, y_pred, average='macro',  zero_division=0)
    f1mi = f1_score(y_te, y_pred, average='micro',  zero_division=0)
    hl   = hamming_loss(y_te, y_pred)
    cv_f1_macro.append(f1ma)
    cv_f1_micro.append(f1mi)
    cv_hamming.append(hl)
    out.append(f"  Fold {fold}: Macro F1={f1ma:.4f}  Micro F1={f1mi:.4f}  Hamming={hl:.6f}")

out.append(f"  MEAN: Macro F1={np.mean(cv_f1_macro):.4f} (+/-{np.std(cv_f1_macro):.4f})")
out.append(f"  MEAN: Micro F1={np.mean(cv_f1_micro):.4f} (+/-{np.std(cv_f1_micro):.4f})")
out.append(f"  MEAN: Hamming ={np.mean(cv_hamming):.6f} (+/-{np.std(cv_hamming):.6f})")

# ── LR Urgency 5-fold CV ──────────────────────────────────────────────
from sklearn.naive_bayes import MultinomialNB
out.append("\n=== 5-FOLD CV: Naive Bayes (urgency, weighted F1) ===")
kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
cv_urg = []
for tr_idx, te_idx in kf2.split(X_all):
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_all[tr_idx], y_urg_all[tr_idx])
    pred = nb.predict(X_all[te_idx])
    cv_urg.append(f1_score(y_urg_all[te_idx], pred, average='weighted', zero_division=0))
out.append(f"  Fold scores: {[round(x,4) for x in cv_urg]}")
out.append(f"  MEAN weighted F1 = {np.mean(cv_urg):.4f} (+/-{np.std(cv_urg):.4f})")

# ════════════════════════════════════════════════════════════════
# 4. SIDE-BY-SIDE COMPARISON
# ════════════════════════════════════════════════════════════════
lr_proba = np.array(lr_clf.predict_proba(X_test_tfidf))
lr_pred  = (lr_proba >= 0.5).astype(int)

out.append("\n=== HEAD-TO-HEAD: LR vs LSTM (120 test samples) ===")
out.append(f"{'Metric':<20} {'LR (Baseline)':>16} {'LSTM':>16}")
out.append("-" * 55)
lr_hl  = hamming_loss(y_issues_test, lr_pred)
lr_sa  = accuracy_score(y_issues_test, lr_pred)
lr_f1m = f1_score(y_issues_test, lr_pred, average='macro',  zero_division=0)
lr_f1i = f1_score(y_issues_test, lr_pred, average='micro',  zero_division=0)

out.append(f"{'Hamming Loss':<20} {lr_hl:>16.6f} {hl:>16.6f}")
out.append(f"{'Subset Accuracy':<20} {lr_sa:>16.4f} {sa:>16.4f}")
out.append(f"{'Macro F1':<20} {lr_f1m:>16.4f} {f1ma:>16.4f}")
out.append(f"{'Micro F1':<20} {lr_f1i:>16.4f} {f1mi:>16.4f}")

out.append("\nPer-class F1 comparison:")
out.append(f"{'Category':<25} {'LR F1':>10} {'LSTM F1':>10} {'Winner':>10}")
out.append("-" * 55)
for i, cat in enumerate(cats):
    _, _, f_lr, s_lr = precision_recall_fscore_support(
        y_issues_test[:, i], lr_pred[:, i], average=None, zero_division=0)
    _, _, f_lstm, _  = precision_recall_fscore_support(
        y_issues_test[:, i], issue_pred_lstm[:, i], average=None, zero_division=0)
    f1_lr_v   = f_lr[1]   if len(f_lr)   > 1 else f_lr[0]
    f1_lstm_v = f_lstm[1] if len(f_lstm) > 1 else f_lstm[0]
    winner = "LR" if f1_lr_v >= f1_lstm_v else "LSTM"
    out.append(f"{cat:<25} {f1_lr_v:>10.4f} {f1_lstm_v:>10.4f} {winner:>10}")

with open('results/lstm_eval_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\n'.join(out))
print("\nDone. Saved to results/lstm_eval_results.txt")
