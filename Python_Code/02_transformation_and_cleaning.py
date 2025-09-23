# Date cleaning and transformation

import pandas as pd

df = pd.read_csv("uber_data_2024.csv")


df['pickup_datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'], errors='coerce')

df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['weekday'] = df['pickup_datetime'].dt.day_name()
df['month'] = df['pickup_datetime'].dt.month_name()

# Replace NaN by 0
canceled_rides = ['Cancelled Rides by Customer', 'Cancelled Rides by Driver', 'Incomplete Rides']
for col in canceled_rides:
    df[col] = df[col].fillna(0)

# Replace NaN by "Not Applicable"
not_applicable = ['Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason']
for col in not_applicable:
    df[col] = df[col].fillna("Not Applicable")

# Replace Distance, fare, time NaN by 0
num_values = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance']
for col in num_values:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Replace Ratings NaN by 0
df['Driver Ratings'] = pd.to_numeric(df['Driver Ratings'], errors='coerce').fillna(0)
df['Customer Rating'] = pd.to_numeric(df['Customer Rating'], errors='coerce').fillna(0)

# Replace Payment Method NaN by "Unknown"
df['Payment Method'] = df['Payment Method'].fillna("Unknown")

# Price per km and per minute
df['fare_per_km'] = df['Booking Value'] / (df['Ride Distance'] + 0.001)
df['fare_per_min'] = df['Booking Value'] / (df['Avg CTAT'] + 0.001)

df = df.drop_duplicates()

df.to_csv("uber_clean_data_2024.csv", index=False)
print("Clean Dataset exported to'uber_clean.csv'")
