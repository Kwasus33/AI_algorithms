import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate, n_iters, threshold):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold

    def _sigmoid(self, x):
        x = np.clip(x)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        x_samples, x_features = X.shape  # (rows, columns)
        self.weights = np.zeros(x_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(
                linear_pred
            )  # linear combination of all features is sigmoid arg

            # derivates of cross entropy loss which we minimize
            dw = (1 / x_samples) * np.dot(X.T, (predictions - y))
            db = (1 / x_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        return np.array([1 if pred > self.threshold else 0 for pred in y_pred])
    
    def predict_prob(self, X):
        return np.zeros(len(X))
