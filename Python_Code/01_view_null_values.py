# View top values to understand null distributions per column

import pandas as pd

def top_values_per_column(df, n):
    result = {}
    for col in df.columns:
        result[col] = df[col].value_counts(dropna=False).head(n)
    return result

df = pd.read_csv("uber_data_2024.csv")

top_vals = top_values_per_column(df, n=5)

for col, vals in top_vals.items():
    print(f"{vals}\n")
