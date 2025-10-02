# Driver Rating Prediction without Data Leakage

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from helpers.analyze_feature_importance import analyze_feature_importance
from helpers.model_evaluation import compare_models, evaluate_classification_model
from helpers.model_pipeline import create_modeling_pipeline

df = pd.read_csv("uber_features_no_leakage_2024.csv")

df_completed = df[df['Booking Status'] == 'Completed'].copy()
df_completed = df_completed.dropna(subset=['High_Driver_Rating'])

features = [
    'hour', 'day', 'month_num', 'pickup_hour', 'pickup_weekday_num',
    'VehicleType_Group', 'Payment Method',
    'Avg VTAT', 'Avg CTAT', 'Ride Distance',
    'fare_per_km', 'fare_per_min'
]

X = df_completed[features].copy()
y = df_completed['High_Driver_Rating']

print(f"Target: High_Driver_Rating")
print(f"Distribution: {y.value_counts(normalize=True).to_dict()}")
print(f"Features: {len(features)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced', C=0.1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=6, random_state=42, class_weight='balanced'
    ), # type: ignore
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', max_depth=8
    )
}

results = {}

for name, model in models.items():
    pipeline = create_modeling_pipeline(X_train, model)
    
    model_results = evaluate_classification_model(
        pipeline, X_train, X_test, y_train, y_test, name
    )
    
    results[name] = model_results

comparison_df = compare_models(results)

analyze_feature_importance(results['Random Forest']['model'], X_train, target_name="Driver Rating",)
