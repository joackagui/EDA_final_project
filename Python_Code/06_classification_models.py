# Classification models for Uber dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("uber_prepared_data_2024.csv")

# Ensure target is numeric (0/1)
if "Is_Cancelled" not in df.columns:
    raise KeyError("Column 'Is_Cancelled' not found in dataset.")
df["Is_Cancelled"] = df["Is_Cancelled"].astype(int)

def frequency_table(column):
    abs_counts = df[column].value_counts(dropna=False)
    rel_counts = df[column].value_counts(normalize=True, dropna=False)
    freq_table = pd.DataFrame({
        "Absolute": abs_counts,
        "Relative": rel_counts.round(4)
    })
    return freq_table

# Print class distribution
categorical_cols = ["Is_Cancelled", "Is_Weekend"]
for col in categorical_cols:
    print(f"\nDistribution of {col}:")
    print(frequency_table(col))

# Drop identifier/text columns not needed for modeling
drop_candidates = [
    "Booking ID", "Customer ID", "Driver ID",
    "Pickup Location", "Drop Location",
    "Reason for cancelling by Customer", "Driver Cancellation Reason",
    "Incomplete Rides Reason", "Date", "Time", "pickup_datetime"
]
to_drop = [c for c in drop_candidates if c in df.columns]
if to_drop:
    print("\nDropping these explicit text/id columns:", to_drop)
    df = df.drop(columns=to_drop, errors="ignore")

# Detect any remaining non-numeric columns and drop them
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("\nFound remaining non-numeric columns (will be dropped):", non_numeric)
    df = df.drop(columns=non_numeric, errors="ignore")
else:
    print("\nNo remaining non-numeric columns found.")

# Check remaining NaNs and basic stats
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
if not nan_counts.empty:
    print("\nColumns with missing values (counts):")
    print(nan_counts)
else:
    print("\nNo missing values detected.")

# Split X, y
X = df.drop(columns=["Is_Cancelled"])
y = df["Is_Cancelled"]

# Sanity: ensure X is numeric
assert X.select_dtypes(exclude=[np.number]).shape[1] == 0, "There are still non-numeric columns in X."

# Train-test split (stratify by y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models with imputation inside pipelines
log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

tree_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
])

models = {
    "Logistic Regression": log_pipeline,
    "Decision Tree": tree_pipeline
}

# Evaluation with K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n {name}")
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print("Error during cross_val_score:", e)
        # continue trying to fit the model to show final evaluation

    # Train final model and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Cancelled", "Cancelled"],
                yticklabels=["Not Cancelled", "Cancelled"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
