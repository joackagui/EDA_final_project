# Descriptive Statistics for Uber Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("uber_clean_data_2024.csv")

# Numerical Summary (Quantity)
num_cols = ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "fare_per_km", "fare_per_min"]
print("\nNumerical Summary")
print(df[num_cols].describe().T) 

# Categorical Summary (Quality)
cat_cols = ["Booking Status", "Vehicle Type", "Payment Method"]
print("\nCategorical Summary")
for col in cat_cols:
    print(f"\nColumn: {col}")
    freq_table = pd.DataFrame({
        "Absolute Frequency": df[col].value_counts(),
    })
    freq_table["Cumulative Absolute Frequency"] = freq_table["Absolute Frequency"].cumsum()
    freq_table["Relative Frequency (%)"] = df[col].value_counts(normalize=True) * 100
    freq_table["Cumulative Relative Frequency(%)"] = freq_table["Relative Frequency (%)"].cumsum()
    print(freq_table)

# Histograms
df[num_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Numerical Variables")
plt.show()

# Boxplots 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f"Boxplot of {col}")
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Variables")
plt.show()
