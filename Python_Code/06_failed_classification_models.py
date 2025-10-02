# Classification models for multiple targets with cross-validation and evaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("uber_prepared_data_2024.csv")

targets = ["Is_Cancelled", "HighValueRide", "CustomerSatisfaction", "VehicleType_Target"]

# Drop leakage-prone columns if still present
leakage_cols = [
    "Cancelled Rides by Customer", "Cancelled Rides by Driver",
    "Incomplete Rides", "Incomplete Rides Reason"
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

# Define preprocessing: numeric vs categorical
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# We will redefine numeric/categorical per target later (to not drop target from X)

# Define models
def make_preprocessor(X):
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_feats),
            ("cat", cat_transformer, cat_feats)
        ]
    )
    return preprocessor

def make_pipelines(X):
    preprocessor = make_preprocessor(X)

    log_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=3000, solver="saga"))
    ])

    tree_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))
    ])

    return {
        "Logistic Regression": log_pipeline,
        "Decision Tree": tree_pipeline
    }

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop through targets
for target in targets:
    print(f"\nTarget: {target}")

    # Define X, y
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical target if needed
    if y.dtype == "object":
        y = y.astype("category")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = make_pipelines(X)

    for name, model in models.items():
        print(f"\n{name}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train final model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confusion matrix plot (skip if too many classes)
        if len(np.unique(y_test)) <= 6:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name} ({target})")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.show()