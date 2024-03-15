from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define the hyperparameters to tune
param_dist = {
    'C': uniform(loc=0, scale=10),  # Regularization strength
    'penalty': ['l1', 'l2'],        # Regularization penalty
    'solver': ['liblinear', 'saga'],# Solver algorithm
    'class_weight': [None, 'balanced'],  # Class weights
    'max_iter': [100, 200, 300]            # Maximum number of iterations
}

# Create a logistic regression model
logistic_model = LogisticRegression()

# Perform randomized search cross-validation
random_search = RandomizedSearchCV(logistic_model, param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model with the best hyperparameters
best_model = random_search.best_estimator_
best_accuracy = best_model.score(X_test, y_test)
print("Best Model Accuracy:", best_accuracy)
