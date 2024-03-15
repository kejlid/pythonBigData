import pandas as pd
import numpy as np

# Load the dataset from the provided URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create DataFrame with features and target
df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
df["PRICE"] = target

# Split the dataset into features (X) and target (y)
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

print(df.head())

# Check the shape of the DataFrame
print("Shape of DataFrame:", df.shape)