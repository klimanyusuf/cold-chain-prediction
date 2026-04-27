import numpy as np
import xgboost as xgb
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=" * 50)
print("TRAINING XGBOOST ANOMALY DETECTION MODEL")
print("=" * 50)

data = np.load("data/processed/xgboost_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"Training on {len(X_train)} samples...")

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ XGBOOST RESULTS:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   AUC:       {auc:.4f}")

joblib.dump(model, "models/xgboost_model.pkl")

# Save ALL metrics including AUC
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "auc": float(auc)
}
with open("models/xgboost_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\n✅ Model saved to: models/xgboost_model.pkl")
print("✅ Metrics saved to: models/xgboost_metrics.json")