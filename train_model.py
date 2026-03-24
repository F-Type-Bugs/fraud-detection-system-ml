import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("creditcard.csv", na_values="?")

# -------------------------
# Feature / Target Split
# -------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Baseline Logistic Regression
# -------------------------
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# print("=== Baseline Model ===")
# print(confusion_matrix(y_test, y_pred_baseline))
# print(classification_report(y_test, y_pred_baseline))

# -------------------------
# Apply SMOTE
# -------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# print("After SMOTE:")
# print(y_train_smote.value_counts())

# -------------------------
# Logistic Regression after SMOTE
# -------------------------
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)

# Fraud probability
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

# -------------------------
# ROC Curve + AUC
# -------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob_smote)
auc_score = roc_auc_score(y_test, y_prob_smote)

# print(f"AUC Score: {auc_score:.3f}")

# -------------------------
# Threshold Tuning
# -------------------------
threshold = 0.3
y_pred_threshold = (y_prob_smote >= threshold).astype(int)

# print("\n=== After SMOTE + Threshold Tuning ===")
# print(confusion_matrix(y_test, y_pred_threshold))
# print(classification_report(y_test, y_pred_threshold))

# -------------------------
# Feature Importance
# -------------------------
importance = pd.Series(
    model_smote.coef_[0],
    index=X.columns
).sort_values(ascending=False)

# print("\nTop 10 Important Features:")
# print(importance.head(10))

# -------------------------
# ROC Curve Plot
# -------------------------
# plt.figure()
# plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
# plt.plot([0, 1], [0, 1], linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate (Recall)")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

# -------------------------
# Export Model
# -------------------------
joblib.dump(model_smote, "fraud_model.pkl")
joblib.dump(X.columns.tolist(), "fraud_feature_columns.pkl")

print("\nModel and feature columns saved successfully.")