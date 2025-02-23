import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, epochs=20000, task_type="classification"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.task_type = task_type
        self.layers = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layers) - 1  # input layer not included
        
        np.random.seed(1)
        self.weights = {}
        self.biases = {}

        # Initialize all the weights and biases
        for i in range(1, len(self.layers)):
            self.weights[f"W{i}"] = np.random.rand(self.layers[i-1], self.layers[i]) - 0.5
            self.biases[f'B{i}'] = np.random.rand(self.layers[i], 1) - 0.5

    def forward_propagation(self, X):
        memo = {"A0": X}
        for i in range(1, self.num_layers):
            memo[f"Z{i}"] = np.dot(self.weights[f"W{i}"].T, memo[f"A{i-1}"]) + self.biases[f"B{i}"]
            memo[f"A{i}"] = np.maximum(0, memo[f"Z{i}"])  # ReLU activation
        memo[f"Z{self.num_layers}"] = np.dot(self.weights[f"W{self.num_layers}"].T, memo[f"A{self.num_layers-1}"]) + self.biases[f"B{self.num_layers}"]
        
        # Activation of output layer
        if self.task_type == "classification":
            memo[f"A{self.num_layers}"] = 1 / (1 + np.exp(-memo[f"Z{self.num_layers}"]))  # Sigmoid
        else:
            memo[f"A{self.num_layers}"] = memo[f"Z{self.num_layers}"]  # Linear
        return memo

    def compute_loss(self, Y_pred, Y):
        m = Y.shape[1]
        if self.task_type == "classification":
            return np.sum(-(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))) / m  # Binary Cross Entropy
        else:
            return np.sum((Y_pred - Y) ** 2) / (2 * m)  # MSE

    def backward_propagation(self, X, Y, memo):
        m = X.shape[1]
        gradients = {}

        # Output layer
        dZ = memo[f"A{self.num_layers}"] - Y
        gradients[f"dW{self.num_layers}"] = np.dot(memo[f"A{self.num_layers-1}"], dZ.T) / m
        gradients[f"dB{self.num_layers}"] = np.sum(dZ, axis=1, keepdims=True) / m

        # Hidden layer
        for i in range(self.num_layers - 1, 0, -1):
            dA = np.dot(self.weights[f"W{i+1}"], dZ)
            dZ = dA * (memo[f"A{i}"] > 0)  # ReLU derivative
            gradients[f"dW{i}"] = np.dot(memo[f"A{i-1}"], dZ.T) / m
            gradients[f"dB{i}"] = np.sum(dZ, axis=1, keepdims=True) / m

        # Update gradient
        for i in range(1, self.num_layers + 1):
            self.weights[f"W{i}"] -= self.learning_rate * gradients[f"dW{i}"]
            self.biases[f"B{i}"] -= self.learning_rate * gradients[f"dB{i}"]

    def train(self, X, Y):
        for epoch in range(self.epochs):
            memo = self.forward_propagation(X)
            loss = self.compute_loss(memo[f"A{self.num_layers}"], Y)
            self.backward_propagation(X, Y, memo)

    def predict(self, X):
        memo = self.forward_propagation(X)
        if self.task_type == "classification":
            threshold = 0.5
            return (memo[f"A{self.num_layers}"] > threshold).astype(int)
        return memo[f"A{self.num_layers}"]