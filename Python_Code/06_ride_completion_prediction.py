# Ride Completion Prediction for Uber Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from helpers.model_pipeline import create_modeling_pipeline
from helpers.analyze_feature_importance import analyze_feature_importance
from helpers.model_evaluation import evaluate_classification_model, compare_models

df = pd.read_csv("uber_features_no_leakage_2024.csv")

# Target variable
df['Ride_Completed'] = (df['Booking Status'] == 'Completed').astype(int)

print(f"Target: Ride_Completed")
print(f"\nDistribution: {df['Ride_Completed'].value_counts(normalize=True).to_dict()}")

# Features
features = [
    'hour', 'day', 'month_num', 'pickup_hour', 'pickup_weekday_num', 'Is_Weekend',
    'VehicleType_Group', 'Payment Method',
    'fare_per_km', 'fare_per_min'
]

X = df[features].copy()
y = df['Ride_Completed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nData: Train {X_train.shape}, Test {X_test.shape}")

# Models
models = {
    'Multiple Linear Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

# Train and evaluate
results = {}
for name, model in models.items():
    pipeline = create_modeling_pipeline(X_train, model)
    results[name] = evaluate_classification_model(
        pipeline, X_train, X_test, y_train, y_test, name
    )

compare_models(results)

importance_df = analyze_feature_importance(results['Random Forest']['model'], X_train, target_name="Driver Rating",)

if importance_df is not None:
    importance_df.to_csv("feature_importances_completion.csv", index=False)
    print("Feature importances saved as feature_importances_completion.csv")