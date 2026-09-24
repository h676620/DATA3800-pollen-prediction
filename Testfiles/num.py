import numpy as np
import pandas as pd

print("--- Data Science Environment Test ---")
print(f"NumPy Version:  {np.__version__}")
print(f"Pandas Version: {pd.__version__}\n")

# 1. Create data using NumPy
dates = pd.date_range("20260101", periods=6)
data = np.random.randn(6, 3)

# 2. Load it into a Pandas DataFrame
df = pd.DataFrame(data, index=dates, columns=["Metric_A", "Metric_B", "Metric_C"])

print("1. Success: Generated Pandas DataFrame using NumPy arrays:")
print(df)

# 3. Test basic Pandas operation (Filtering / Aggregation)
mean_a = df["Metric_A"].mean()
print(f"\n2. Success: Computed mean of Metric_A: {mean_a:.4f}")

print("\nEnvironment check passed successfully!")
