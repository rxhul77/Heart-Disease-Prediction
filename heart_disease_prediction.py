import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("heart_disease_uci.csv")

print("Original dataset shape:", df.shape)

# Remove ID column
df = df.drop(columns=["id"])

# Convert num into binary target
# 0 = No Heart Disease
# 1,2,3,4 = Heart Disease
df["target"] = (df["num"] > 0).astype(int)

# Remove original num column
df = df.drop(columns=["num"])

print("\nTarget distribution:")
print(df["target"].value_counts())

# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = df.drop(columns=["target"])
y = df["target"]

# --------------------------------------------------
# 3. Define Feature Types
# --------------------------------------------------

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]

categorical_features = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 6. Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 7. Decision Tree
# --------------------------------------------------

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])

decision_tree_model.fit(X_train, y_train)

tree_predictions = decision_tree_model.predict(X_test)
tree_probabilities = decision_tree_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

print("\n========== LOGISTIC REGRESSION ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, logistic_predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, logistic_predictions), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, logistic_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\n========== DECISION TREE ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, tree_predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, tree_predictions), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, tree_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_predictions))

# --------------------------------------------------
# 9. Save Logistic Regression Confusion Matrix
# --------------------------------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    "confusion_matrix_logistic.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 10. Save Decision Tree Confusion Matrix
# --------------------------------------------------

tree_cm = confusion_matrix(
    y_test,
    tree_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=tree_cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()

plt.savefig(
    "confusion_matrix_decision_tree.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 11. ROC Curve and AUC
# --------------------------------------------------

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

tree_fpr, tree_tpr, _ = roc_curve(
    y_test,
    tree_probabilities
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

tree_auc = roc_auc_score(
    y_test,
    tree_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label=f"Decision Tree (AUC = {tree_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Heart Disease Prediction")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 12. Print AUC
# --------------------------------------------------

print("\n========== ROC-AUC ==========")

print(
    "Logistic Regression AUC:",
    round(logistic_auc, 4)
)

print(
    "Decision Tree AUC:",
    round(tree_auc, 4)
)

print("\nAll graphs generated successfully!")
# --------------------------------------------------
# 13. Save Model Comparison
# --------------------------------------------------

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy_score(y_test, logistic_predictions),
        accuracy_score(y_test, tree_predictions)
    ],
    "Precision": [
        precision_score(y_test, logistic_predictions),
        precision_score(y_test, tree_predictions)
    ],
    "Recall": [
        recall_score(y_test, logistic_predictions),
        recall_score(y_test, tree_predictions)
    ],
    "AUC": [
        logistic_auc,
        tree_auc
    ]
})

model_comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved as model_comparison.csv")

# --------------------------------------------------
# 14. Save Predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "Actual": y_test.values,

    "Logistic_Regression_Prediction":
        logistic_predictions,

    "Logistic_Regression_Probability":
        logistic_probabilities,

    "Decision_Tree_Prediction":
        tree_predictions,

    "Decision_Tree_Probability":
        tree_probabilities
})

predictions_df.to_csv(
    "predictions.csv",
    index=False
)

print("Predictions saved as predictions.csv")