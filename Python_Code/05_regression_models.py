# Regression Models for Uber Dataset

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv("uber_prepared_data_2024.csv")

# Keep only numeric columns (drop object/text automatically)
df = df.select_dtypes(include=["number"])

# Define features and target
X = df.drop(columns=["Booking Value"])
y = df["Booking Value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    ),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n {name}")
    # Cross-validation R²
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    print(f"Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))
