# Visualizations for Ride Completion Prediction and Business Insights

import pandas as pd
from helpers.visualizations_completion import plot_revenue_impact, plot_ride_completion_distribution, plot_feature_importance
from helpers.visualizations_business import plot_peak_hours, plot_cancellation_reasons, plot_model_performance

df = pd.read_csv("uber_features_no_leakage_2024.csv")

feature_importance = {
    'fare_per_min': 0.28,
    'fare_per_km': 0.27,
    'Payment Method_Unknown': 0.23,
    'Payment Method_UPI': 0.04,
    'day': 0.03,
    'month_num': 0.02,
    'hour': 0.02,
    'pickup_hour': 0.02,
    'Payment Method_Cash': 0.02,
    'pickup_weekday_num': 0.01
    }

# Plots
plot_peak_hours(df)
plot_cancellation_reasons(df)
plot_ride_completion_distribution(df)
plot_revenue_impact(df)
plot_model_performance()

plot_feature_importance(feature_importance)