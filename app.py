import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Generate a dataset with random values
# Generate features with a normal distribution (1000 samples, 4 features)
X = np.random.normal(0, 1, size=(1000, 4))  # Mean=0, Std=1, 4 features

# Generate target labels (continuous values for regression)
y = np.random.normal(0, 1, size=1000)  # Continuous target labels

# Split dataset into training and test set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model using the training data
regressor.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = regressor.predict(X_test)

# Calculate and print Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(regressor, 'linear_regression_model.pkl')
print("Model saved!")
