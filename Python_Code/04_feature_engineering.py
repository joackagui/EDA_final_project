# Feature Engineering for Uber Dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("uber_clean_data_2024.csv")

# Convert to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

# Extract features from Date
df["year"] = df["Date"].dt.year
df["month_num"] = df["Date"].dt.month
df["day_num"] = df["Date"].dt.day

# Extract features from pickup_datetime
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day"] = df["pickup_datetime"].dt.day
df["pickup_weekday_num"] = df["pickup_datetime"].dt.weekday
df["pickup_month_num"] = df["pickup_datetime"].dt.month

# New engineered features
df["Is_Cancelled"] = df["Booking Status"].apply(lambda x: 1 if "Cancelled" in str(x) else 0)
df["Is_Weekend"] = df["pickup_weekday_num"].apply(lambda x: 1 if x >= 5 else 0)  # Sat=5, Sun=6
df["value_per_km"] = df["Booking Value"] / (df["Ride Distance"] + 1e-5)

# Encode categorical variables
cat_cols = ["Booking Status", "Vehicle Type", "Payment Method", "weekday", "month"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical features
num_cols = ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "fare_per_km", "fare_per_min"]
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Drop original raw columns
drop_cols = ["Date", "Time", "pickup_datetime"]
df_encoded = df_encoded.drop(columns=drop_cols, errors="ignore")

# Drop raw and identifier columns
drop_cols = ["Date", "Time", "pickup_datetime", "Booking ID", "Customer ID", "Driver ID"]
df_encoded = df_encoded.drop(columns=drop_cols, errors="ignore")

# Save prepared dataset
df_encoded.to_csv("uber_prepared_data_2024.csv", index=False)


print("Success. Saved as uber_prepared_data_2024.csv")
print("New features: year, month_num, day_num, pickup_hour, pickup_day, pickup_weekday_num, pickup_month_num, Is_Cancelled, Is_Weekend, value_per_km")
