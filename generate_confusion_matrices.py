
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Save files directly in the project folder
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "heart_disease_uci.csv"

df = pd.read_csv(CSV_PATH)

# Prepare target
df = df.drop(columns=["id"])
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

X = df.drop(columns=["target"])
y = df["target"]

# Feature columns
numerical_features = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca"
]

categorical_features = [
    "sex", "dataset", "cp", "fbs",
    "restecg", "exang", "slope", "thal"
]

# Preprocessing
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

# Same train/test split as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "confusion_matrix_logistic.png"
    ),
    "Decision Tree": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=42,
                max_depth=5
            ))
        ]),
        "confusion_matrix_decision_tree.png"
    )
}

for model_name, (model, filename) in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()

    output_path = BASE_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {output_path}")
    print(cm)

print("\nFinished generating both confusion matrix images.")