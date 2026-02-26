"""
Full analysis export: raw predictions, threshold behavior, ablation, coverage-risk, confusion matrix.
"""
import pickle, json, numpy as np, csv, os
from scipy import sparse
from sklearn.metrics import (
    hamming_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
)

X_test_tfidf = sparse.load_npz('data/X_test_tfidf.npz')
y_issues_test = np.load('data/y_issues_test.npy')    # (120, 8)
y_urgency_test = np.load('data/y_urgency_test.npy')  # (120,)

with open('data/category_mapping.json') as f:
    mappings = json.load(f)
categories = mappings['categories']  # 8 classes
urgency_labels = ['low', 'medium', 'high']

with open('models/issue_classifier_ml.pkl', 'rb') as f:
    issue_clf = pickle.load(f)
with open('models/urgency_classifier_ml.pkl', 'rb') as f:
    urgency_clf = pickle.load(f)

issue_proba = np.array(issue_clf.predict_proba(X_test_tfidf))  # (120, 8)
print(f"issue_proba shape: {issue_proba.shape}")

urgency_proba = np.array(urgency_clf.predict_proba(X_test_tfidf))  # (120, 3)
urgency_pred  = urgency_clf.predict(X_test_tfidf)                  # (120,)
issue_pred_05 = (issue_proba >= 0.5).astype(int)

os.makedirs('results', exist_ok=True)

# ── 1. Raw Predictions CSV ────────────────────────────────────────────
with open('results/raw_test_predictions.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    hdr  = ['sample_id', 'true_labels']
    hdr += [f'prob_{c}' for c in categories]
    hdr += [f'pred_{c}' for c in categories]
    hdr += ['true_urgency', 'prob_low', 'prob_medium', 'prob_high', 'pred_urgency']
    w.writerow(hdr)
    for i in range(120):
        true_lbls = '|'.join(categories[j] for j in range(8) if y_issues_test[i, j] == 1)
        row  = [i, true_lbls]
        row += [round(float(issue_proba[i, j]), 6) for j in range(8)]
        row += [int(issue_pred_05[i, j]) for j in range(8)]
        row += [urgency_labels[y_urgency_test[i]]]
        row += [round(float(urgency_proba[i, k]), 6) for k in range(3)]
        row += [urgency_labels[urgency_pred[i]]]
        w.writerow(row)
print("Saved: results/raw_test_predictions.csv")

# ── 2. Threshold Behavior ─────────────────────────────────────────────
def route_stats(th_hi, th_lo):
    ar = mr = ig = aw = 0
    for i in range(120):
        mc = float(issue_proba[i].max())
        bi = int(issue_proba[i].argmax())
        if mc >= th_hi:
            ar += 1
            if y_issues_test[i, bi] != 1:
                aw += 1
        elif mc >= th_lo:
            mr += 1
        else:
            ig += 1
    ac  = ar - aw
    er  = aw / ar * 100 if ar > 0 else float('nan')
    cov = ar / 120 * 100
    return ar, mr, ig, ac, aw, er, cov

print("\nTHRESHOLD BEHAVIOR (n=120 test samples)")
print(f"{'Scenario':<14} | {'AutoRoute':>10} | {'Review':>9} | {'Ignored':>9} | {'Correct':>9} | {'Wrong':>7} | {'ErrorRate':>10}")
print("-" * 80)
for th_hi, th_lo in [(0.85, 0.65), (0.65, 0.50), (0.50, 0.00)]:
    ar, mr, ig, ac, aw, er, cov = route_stats(th_hi, th_lo)
    scn = f"{th_hi:.0%}/{th_lo:.0%}"
    er_str = f"{er:6.2f}%" if not (isinstance(er, float) and np.isnan(er)) else "  N/A  "
    print(f"{scn:<14} | {ar:3d}({cov:5.1f}%) | {mr:3d}({mr/120*100:4.1f}%) | {ig:3d}({ig/120*100:4.1f}%) | {ac:9d} | {aw:7d} | {er_str:>10}")

# Argmax (always route, no abstention)
ac2 = aw2 = 0
for i in range(120):
    bi = int(issue_proba[i].argmax())
    if y_issues_test[i, bi] == 1: ac2 += 1
    else: aw2 += 1
print(f"{'Argmax':<14} | 120(100%)  |   0( 0.0%) |   0( 0.0%) | {ac2:9d} | {aw2:7d} | {aw2/120*100:9.2f}%")

# ── 3. Ablation Table ─────────────────────────────────────────────────
print("\nABLATION TABLE")
print(f"{'Strategy':<35} | {'Coverage':>10} | {'Precision':>10} | {'Misrouted':>10}")
print("-" * 70)
# Argmax
print(f"{'Pure Argmax (no threshold)':<35} | {'100.0%':>10} | {ac2/120*100:>9.2f}% | {aw2:>10d}")
# 85% threshold auto-route only
ar85, _, _, ac85, aw85, er85, cov85 = route_stats(0.85, 0.00)
p85 = ac85 / ar85 * 100 if ar85 > 0 else float('nan')
print(f"{'Auto-route @ 85% (no review)':<35} | {cov85:>9.1f}% | {p85:>9.2f}% | {aw85:>10d}")
# 65% threshold auto-route only
ar65, _, _, ac65, aw65, er65, cov65 = route_stats(0.65, 0.00)
p65 = ac65 / ar65 * 100 if ar65 > 0 else float('nan')
print(f"{'Auto-route @ 65% (no review)':<35} | {cov65:>9.1f}% | {p65:>9.2f}% | {aw65:>10d}")

# ── 4. Per-class Precision / Recall / F1 ─────────────────────────────
print("\nPER-CLASS METRICS  (Issue LR, threshold=0.5, test set)")
print(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 65)
for i, cat in enumerate(categories):
    p, r, f, s = precision_recall_fscore_support(
        y_issues_test[:, i], issue_pred_05[:, i], average=None, zero_division=0
    )
    if len(p) > 1:
        print(f"{cat:<25} {p[1]:>10.4f} {r[1]:>10.4f} {f[1]:>10.4f} {int(s[1]):>10d}")
    else:
        print(f"{cat:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {int(s[0]):>10d}")

# ── 5. Urgency Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_urgency_test, urgency_pred)
print("\nURGENCY CONFUSION MATRIX  (Naive Bayes, test set)")
print(f"{'':12}", end='')
for lbl in urgency_labels:
    print(f"{'Pred_'+lbl:>14}", end='')
print()
for i, lbl in enumerate(urgency_labels):
    print(f"{'True_'+lbl:12}", end='')
    for j in range(3):
        print(f"{cm[i, j]:>14d}", end='')
    print()

# ── 6. Hamming Loss vs Threshold ─────────────────────────────────────
print("\nHAMMING LOSS vs THRESHOLD  (issue classifier, test set)")
print(f"{'Threshold':>12} | {'HammingLoss':>14} | {'SubsetAcc':>12}")
print("-" * 44)
for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.85]:
    pred = (issue_proba >= t).astype(int)
    hl = hamming_loss(y_issues_test, pred)
    sa = accuracy_score(y_issues_test, pred)
    print(f"{t:>12.2f} | {hl:>14.6f} | {sa:>12.4f}")

# ── 7. Coverage-Risk CSV ──────────────────────────────────────────────
thresholds = np.arange(0.10, 1.00, 0.02)
with open('results/coverage_risk_curve.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['threshold', 'coverage_pct', 'precision_pct', 'risk_pct', 'n_routed'])
    for t in thresholds:
        routed = correct = 0
        for i in range(120):
            if float(issue_proba[i].max()) >= t:
                routed += 1
                if y_issues_test[i, int(issue_proba[i].argmax())] == 1:
                    correct += 1
        cov  = routed / 120 * 100
        prec = correct / routed * 100 if routed > 0 else ''
        risk = (routed - correct) / routed * 100 if routed > 0 else ''
        w.writerow([round(float(t), 2), round(cov, 2),
                    round(prec, 4) if prec != '' else '',
                    round(risk, 4) if risk != '' else '',
                    routed])
print("\nSaved: results/coverage_risk_curve.csv")
print("\nDONE")
