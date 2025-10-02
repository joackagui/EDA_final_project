# Feature Engineering WITHOUT data leakage

import pandas as pd

df = pd.read_csv("uber_clean_data_2024.csv")
df = df.copy()

# Date features (available at booking time)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

# Temporal features
df["year"] = df["Date"].dt.year
df["month_num"] = df["Date"].dt.month
df["day_num"] = df["Date"].dt.day
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day"] = df["pickup_datetime"].dt.day
df["pickup_weekday_num"] = df["pickup_datetime"].dt.weekday
df["pickup_month_num"] = df["pickup_datetime"].dt.month

df["Is_Weekend"] = (df["pickup_weekday_num"] >= 5).astype(int)

# Vehicle type
df["VehicleType_Group"] = df["Vehicle Type"]

print("Features created without data leakage")

# Target 1: High Driver Rating (for completed rides only)
completed_rides = df[df['Booking Status'] == 'Completed'].copy()
if len(completed_rides) > 0 and completed_rides['Driver Ratings'].nunique() > 1:
    rating_threshold = completed_rides['Driver Ratings'].quantile(0.75)
    df['High_Driver_Rating'] = (df['Driver Ratings'] >= rating_threshold).astype(int)
    # Replace not completed rides with NaN
    df.loc[df['Booking Status'] != 'Completed', 'High_Driver_Rating'] = None

# Target 2: Ride Completion
df['Ride_Completed'] = (df['Booking Status'] == 'Completed').astype(int)

# Target 3: Cancellation by Driver
df['Cancelled_by_Driver'] = (df['Booking Status'] == 'Cancelled by Driver').astype(int)

print("Target variables prepared")

df.to_csv("uber_features_no_leakage_2024.csv", index=False)
print("Saved as: uber_features_no_leakage_2024.csv")
