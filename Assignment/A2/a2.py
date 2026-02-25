import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []
    
    # Make a prediction using a trained model
    def predict(self, X):
        # Compute weighted sum + bias
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply activation function
        return np.where(linear_output >= 0, 1, 0)
    
    # Fit the model to the data
    def fit(self, X, y):
        # Get rows and columns of data
        n_samples, n_features = X.shape
        # Set weight for each feature to 0
        self.weights = np.zeros(n_features)
        # Set bias to 0
        self.bias = 0.0
        # Loop through epochs
        for _ in range(self.epochs):
            errors = 0
            # Loop through each input 
            for xi, target in zip(X, y):
                # Calculate weighted sum + bias
                linear_output = np.dot(xi, self.weights) + self.bias
                # Apply activation function
                y_pred = 1 if linear_output >= 0 else 0
                # Compute error term
                update = self.lr * (target - y_pred)
                # Update weights and bias with error term
                self.weights += update * xi
                self.bias += update
                # track incorrect predictions
                errors += int(update != 0)
            # track total incorrect predictions per epoch
            self.errors_per_epoch.append(errors)