# AI Engineering Internship - Classification Project

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

df = X.copy()
df["target"] = y

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget classes:")
print(data.target_names)

print("\nMissing values:")
print(df.isnull().sum().sum())


# ============================================================
# 2. BASIC EDA
# ============================================================

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(df.describe())

print("\nClass distribution:")
print(df["target"].value_counts())


# Target distribution graph
df["target"].value_counts().plot(kind="bar")

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(
    ticks=[0, 1],
    labels=["Malignant", "Benign"],
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 4. CREATE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}


# ============================================================
# 5. TRAIN & EVALUATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\n{name}")
    print("-" * 40)
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=data.target_names
    ))


# ============================================================
# 6. COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

print(results_df.sort_values(
    by="Accuracy",
    ascending=False
))


# Accuracy graph
results_df.plot(
    x="Model",
    y="Accuracy",
    kind="bar",
    legend=False
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# ============================================================
# 7. CROSS VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    print(
        f"{name}: "
        f"Mean CV Accuracy = {scores.mean():.4f}"
    )


# ============================================================
# 8. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_, 4))


# ============================================================
# 9. FINAL MODEL EVALUATION
# ============================================================

best_model = grid_search.best_estimator_

final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

print("Final Test Accuracy:", round(final_accuracy, 4))

print("\nFinal Classification Report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=data.target_names
))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()

plt.title("Confusion Matrix - Final Model")
plt.tight_layout()
plt.show()



# ============================================================
# 11. ROC-AUC SCORE
# ============================================================

from sklearn.metrics import roc_auc_score, roc_curve

probabilities = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n" + "=" * 60)
print("ROC-AUC SCORE")
print("=" * 60)

print("ROC-AUC:", round(auc, 4))


# ROC Curve
fpr, tpr, _ = roc_curve(
    y_test,
    probabilities
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

import joblib

joblib.dump(
    best_model,
    "best_model.pkl"
)

print("\nBest model saved as: best_model.pkl")


# ============================================================
# 11. SAMPLE PREDICTION
# ============================================================

sample = X_test.iloc[[0]]

prediction = best_model.predict(sample)

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

print("Predicted Class:", data.target_names[prediction[0]])
print("Actual Class:", data.target_names[y_test.iloc[0]])

print("\nPROJECT COMPLETED SUCCESSFULLY!")
