# Feature Engineering for Uber Dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("uber_clean_data_2024.csv")

# Date & time processing
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

# Binary cancellation flag
df["Is_Cancelled"] = df["Booking Status"].apply(
    lambda x: 1 if "Cancelled" in str(x) else 0
)

# Weekend indicator
df["Is_Weekend"] = df["pickup_weekday_num"].apply(lambda x: 1 if x >= 5 else 0)

# Value per km
df["value_per_km"] = df["Booking Value"] / (df["Ride Distance"] + 1e-5)

# High-value ride (business logic: top 25% of fares)
fare_threshold = df["Booking Value"].quantile(0.75)
df["HighValueRide"] = (df["Booking Value"] >= fare_threshold).astype(int)

# Customer satisfaction: rating >= 4.5
df["CustomerSatisfaction"] = (df["Customer Rating"] >= 4.5).astype(int)

# Vehicle type target (for classification later)
df["VehicleType_Target"] = df["Vehicle Type"]

# One-hot encode categorical variables
cat_cols = ["Vehicle Type", "Payment Method", "weekday", "month"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical features
num_cols = [
    "Booking Value", "Ride Distance",
    "Driver Ratings", "Customer Rating",
    "fare_per_km", "fare_per_min"
]
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Drop leakage-prone or ID columns
drop_cols = [
    "Date", "Time", "pickup_datetime",
    "Booking ID", "Customer ID", "Driver ID",
    "Pickup Location", "Drop Location",
    "Reason for cancelling by Customer",
    "Driver Cancellation Reason",
    "Incomplete Rides Reason",
    # Booking status dummies (directly reveal Is_Cancelled!)
    "Booking Status"
]
df_encoded = df_encoded.drop(columns=drop_cols, errors="ignore")

# Save prepared dataset
df_encoded.to_csv("uber_prepared_data_2024.csv", index=False)

print("Saved as uber_prepared_data_2024.csv")
