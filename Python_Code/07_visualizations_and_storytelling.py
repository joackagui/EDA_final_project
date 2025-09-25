# Visualizations & Storytelling for Uber Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("uber_prepared_data_2024.csv")

# Cancelled vs Completed rides
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Is_Cancelled", hue="Is_Cancelled", palette="viridis", legend=False)
plt.title("Distribution of Cancelled vs Non-Cancelled Rides")
plt.xticks([0,1], ["Not Cancelled", "Cancelled"])
plt.ylabel("Number of Rides")
plt.show()

# Cancelled rides by month
cancel_by_month = df.groupby("month_num")["Is_Cancelled"].mean() * 100
cancel_by_month = cancel_by_month.sort_index()
plt.figure(figsize=(8,5))
cancel_by_month.plot(kind="bar", color="tomato")
plt.title("Cancellation Rate (%) by Month")
plt.ylabel("Cancellation Rate (%)", fontsize=12)
plt.xlabel("Month", fontsize=12)
plt.xticks(rotation=45)

# Relationship: Distance vs Fare per km
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Ride Distance", y="Booking Value", alpha=0.3)
plt.title("Ride Distance vs Booking Value")
plt.xlabel("Ride Distance (km)")
plt.ylabel("Booking Value")
plt.show()

# Correlation Heatmap
num_cols = ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "fare_per_km", "fare_per_min"]
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Variables")
plt.show()

# Revenue by Vehicle Type (availables in uber_clean_data_2024.csv)
df_clean = pd.read_csv("uber_clean_data_2024.csv")

if "Vehicle Type" in df_clean.columns:
    revenue_by_vehicle = df_clean.groupby("Vehicle Type")["Booking Value"].sum().sort_values(ascending=False)
    revenue_by_vehicle.plot(kind="bar", figsize=(8,5), color="skyblue")
    plt.title("Total Revenue by Vehicle Type")
    plt.ylabel("Revenue")
    plt.xlabel("Vehicle Type")
    plt.show()
