"""
07_statistical_analysis.py - Statistical Depth Enhancements
Run this once after training models. Does not modify existing code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STATISTICAL DEPTH ANALYSIS")
print("=" * 60)

# Load your trained model and data
model = joblib.load("models/xgboost_model.pkl")
data = np.load("data/processed/xgboost_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"Test set size: {len(y_test)} samples")
print(f"Failures in test: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print()

# ============================================================
# 1. CONFIDENCE INTERVALS (Bootstrap with error handling)
# ============================================================
print("1. Computing Confidence Intervals...")

n_bootstrap = 500  # Reduced for speed
accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_scores = []

valid_bootstraps = 0

for i in range(n_bootstrap):
    # Sample with replacement
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    y_test_boot = y_test[idx]
    
    # Skip if only one class present
    if len(np.unique(y_test_boot)) < 2:
        continue
    
    y_pred_boot = model.predict(X_test[idx])
    y_proba_boot = model.predict_proba(X_test[idx])[:, 1]
    
    accuracies.append(accuracy_score(y_test_boot, y_pred_boot))
    precisions.append(precision_score(y_test_boot, y_pred_boot, zero_division=0))
    recalls.append(recall_score(y_test_boot, y_pred_boot, zero_division=0))
    f1_scores.append(f1_score(y_test_boot, y_pred_boot, zero_division=0))
    auc_scores.append(roc_auc_score(y_test_boot, y_proba_boot))
    valid_bootstraps += 1

print(f"Valid bootstraps: {valid_bootstraps}/{n_bootstrap}")

if valid_bootstraps > 0:
    print(f"\n95% Confidence Intervals:")
    print(f"  Accuracy:  [{np.percentile(accuracies, 2.5):.3f}, {np.percentile(accuracies, 97.5):.3f}]")
    print(f"  Precision: [{np.percentile(precisions, 2.5):.3f}, {np.percentile(precisions, 97.5):.3f}]")
    print(f"  Recall:    [{np.percentile(recalls, 2.5):.3f}, {np.percentile(recalls, 97.5):.3f}]")
    print(f"  F1 Score:  [{np.percentile(f1_scores, 2.5):.3f}, {np.percentile(f1_scores, 97.5):.3f}]")
    print(f"  AUC:       [{np.percentile(auc_scores, 2.5):.3f}, {np.percentile(auc_scores, 97.5):.3f}]")

# Save to JSON
ci_results = {
    "accuracy_ci": [float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))] if valid_bootstraps > 0 else [0,0],
    "precision_ci": [float(np.percentile(precisions, 2.5)), float(np.percentile(precisions, 97.5))] if valid_bootstraps > 0 else [0,0],
    "recall_ci": [float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))] if valid_bootstraps > 0 else [0,0],
    "f1_ci": [float(np.percentile(f1_scores, 2.5)), float(np.percentile(f1_scores, 97.5))] if valid_bootstraps > 0 else [0,0],
    "auc_ci": [float(np.percentile(auc_scores, 2.5)), float(np.percentile(auc_scores, 97.5))] if valid_bootstraps > 0 else [0,0]
}
with open("models/confidence_intervals.json", "w") as f:
    json.dump(ci_results, f, indent=2)

# ============================================================
# 2. CROSS-VALIDATION
# ============================================================
print("\n2. Performing 5-Fold Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print(f"5-Fold CV F1 Scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save
cv_results = {
    "f1_scores": cv_scores.tolist(),
    "mean": float(cv_scores.mean()),
    "std": float(cv_scores.std())
}
with open("models/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

# ============================================================
# 3. PRECISION-RECALL CURVE (Better for Imbalanced Data)
# ============================================================
print("\n3. Computing Precision-Recall Curve...")

y_proba = model.predict_proba(X_test)[:, 1]
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_curve, precision_curve)
avg_precision = average_precision_score(y_test, y_proba)

print(f"PR-AUC: {pr_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, color='darkorange', label=f'PR-AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (XGBoost)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: models/pr_curve.png")

# ============================================================
# 4. ROC CURVE
# ============================================================
print("\n4. Computing ROC Curve...")

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, color='darkorange', label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (XGBoost)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: models/roc_curve.png")

# ============================================================
# 5. LEARNING CURVE (Simplified)
# ============================================================
print("\n5. Computing Learning Curve...")

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
train_scores = []
val_scores = []

for size in train_sizes:
    n_samples = int(len(X_train) * size)
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]
    
    # Train a small model for this subset
    from sklearn.ensemble import RandomForestClassifier
    temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
    temp_model.fit(X_subset, y_subset)
    
    y_pred_train = temp_model.predict(X_subset)
    y_pred_val = temp_model.predict(X_test)
    
    train_scores.append(f1_score(y_subset, y_pred_train, zero_division=0))
    val_scores.append(f1_score(y_test, y_pred_val, zero_division=0))

plt.figure(figsize=(8, 6))
plt.plot([s*100 for s in train_sizes], train_scores, 'o-', label='Training F1', linewidth=2)
plt.plot([s*100 for s in train_sizes], val_scores, 'o-', label='Validation F1', linewidth=2)
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: models/learning_curve.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)
print("\nFiles saved in 'models/' folder:")
print("  - confidence_intervals.json")
print("  - cv_results.json")
print("  - pr_curve.png")
print("  - roc_curve.png")
print("  - learning_curve.png")