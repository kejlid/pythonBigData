import pandas as pd
from sklearn.datasets import fetch_openml

# Load the Boston Housing dataset from an alternative source
housing = fetch_openml(name="house_prices", as_frame=True)
df = pd.concat([housing.data, housing.target], axis=1)
df.columns = housing.feature_names + ['MEDV']  # MEDV is the target variable (house prices)

# Compute descriptive statistics
statistics = df.describe()

# Print the summary statistics
print(statistics)
