import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset from the provided URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create DataFrame with features and target
df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
df["PRICE"] = target

# Set a threshold to classify house prices as high or low
threshold = df["PRICE"].median()
df["PRICE_CLASS"] = (df["PRICE"] >= threshold).astype(int)

# Output some rows to verify the dataframe
print("Logistic Regression Model:")
print(df.head())

# Split the dataset into features (X) and target (y) for logistic regression
X_logistic = df.drop(["PRICE", "PRICE_CLASS"], axis=1)
y_logistic = df["PRICE_CLASS"]

# Split the dataset into features (X) and target (y) for linear regression
X_linear = df.drop("PRICE", axis=1)
y_linear = df["PRICE"]

# Split the dataset into training and testing sets for logistic regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

# Split the dataset into training and testing sets for linear regression
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

# Make predictions for logistic regression
y_pred_logistic = logistic_model.predict(X_test_logistic)

# Calculate accuracy for logistic regression
accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)
print("Accuracy (Logistic Regression):", accuracy_logistic)

# Train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)

# Make predictions for linear regression
y_pred_linear = linear_model.predict(X_test_linear)

# Calculate Mean Squared Error for linear regression
mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
print("Mean Squared Error (Linear Regression):", mse_linear)

# Compare the models
if accuracy_logistic > 0.5:
    print("The Logistic Regression model is better for this classification task.")
else:
    print("The Linear Regression model is better for this regression task.")
