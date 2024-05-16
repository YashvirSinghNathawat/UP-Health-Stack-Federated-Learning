import numpy as np

"""
    (i) This implementation of the Support Vector Machine (SVM) algorithm only classifies linear separable data nicely,
     the kernel trick has not been implemented...because in the federated setting we can't share the support vectors to the server, 
        which are the key to the kernel trick...although the kernel trick can be implemented 
        in a secure multi-party computation setting but this problem has some other obvious challenges.
"""


class LinearRegression:
    def __init__(self, C=1.0, gamma='auto', lr=0.01, n_iters=100):
        self.C = C
        self.gamma = gamma
        self.weights = None
        self.biases = None
        self.lr = lr
        self.n_iters = n_iters
        self.is_binary = False

    def fit_binary(self, X, y):
        self.is_binary = True
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        lr = self.lr
        n_iters = self.n_iters
        binary_y = np.where(y == 1, 1, -1)
        for epoch in range(n_iters):
            for idx, x in enumerate(X):
                decision = np.dot(x, weights) + bias
                if binary_y[idx] * decision < 1:
                    weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                    bias += lr * binary_y[idx]
                else:
                    weights += lr * (-2 * self.C * weights)
        self.weights = weights
        self.biases = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        if n_classes == 2:
            self.fit_binary(X, y)
            return

        if self.weights is None or self.biases is None:
            self.weights = np.zeros((n_classes, n_features))
            self.biases = np.zeros(n_classes)
        # print("weights shape", self.weights.shape)
        for i in range(n_classes):
            binary_y = np.where(y == i, 1, -1)
            weights = self.weights[i]
            bias = self.biases[i]
            lr = self.lr  # Learning rate
            n_iters = self.n_iters  # Number of n_iters
            for epoch in range(n_iters):
                for idx, x in enumerate(X):
                    decision = np.dot(x, weights) + bias
                    if binary_y[idx] * decision < 1:
                        weights += lr * (binary_y[idx] * x - 2 * self.C * weights)
                        bias += lr * binary_y[idx]
                    else:
                        weights += lr * (-2 * self.C * weights)
            self.weights[i] = weights
            self.biases[i] = bias

    def predict(self, X):
        if self.weights is None or self.biases is None:
            raise ValueError("Model has not been trained yet.")
        decision_values = np.dot(X, self.weights.T) + self.biases

        if self.is_binary:
            return np.where(decision_values < 0, 0, 1)
        return np.argmax(decision_values, axis=1)

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return self.weights

    def get_biases(self):
        if self.biases is None:
            raise ValueError("Model has not been trained yet.")
        return self.biases

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_biases(self, new_biases):
        self.biases = new_biases
