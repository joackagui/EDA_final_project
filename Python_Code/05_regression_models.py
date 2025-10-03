# Regression Models to Predict Booking Value

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from helpers.regression_pipeline import create_regression_pipeline

df = pd.read_csv("uber_features_no_leakage_2024.csv")

# Filter only completed rides
df_completed = df[df['Booking Status'] == 'Completed'].copy()
print(f"Rides completed: {len(df_completed)}")

# Define features and target
features = [
    'hour', 'day', 'month_num', 'pickup_hour', 'pickup_weekday_num', 'Is_Weekend',
    'VehicleType_Group', 'Payment Method',
    'Ride Distance',
    'fare_per_km', 'fare_per_min'
]

X = df_completed[features].copy()
y = df_completed['Booking Value']

print(f"Data: {X.shape}")
print(f"Target: Booking Value (mean: {y.mean():.2f}, std: {y.std():.2f})")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Models
models = {
    "Linear Regression": create_regression_pipeline(LinearRegression(), X_train),
    "Random Forest": create_regression_pipeline(
        RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10), X_train
    )
}

# Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRegression Model Evaluation")

for name, model in models.items():
    print(f"\n{name}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    print(f"   CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   Test R²: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Interpretation
    if r2 > 0.7:
        print("Excellent for prediction")
    elif r2 > 0.5:
        print("Good for prediction") 
    elif r2 > 0.3:
        print("Moderate for prediction")
    else:
        print("Limited for prediction")
