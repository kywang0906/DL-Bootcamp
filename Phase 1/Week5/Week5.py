import numpy as np

class Network:
    def __init__(self, weights, biases, activation_funcs):
        self.weights = weights
        self.biases = biases
        self.activation_funcs = activation_funcs
        self.grad_weights = [np.zeros_like(w) for w in weights]
        self.grad_biases = [np.zeros_like(b) for b in biases]

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)  # Return 1 where x > 0, else 0
        return np.maximum(0, x) 

    def activation(self, x, name, derivative=False):
        if name == "linear":
            return x if not derivative else np.ones_like(x)
        elif name == "relu":
            return self.relu(x, derivative)
        elif name == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig if not derivative else sig * (1 - sig)
    
    def forward(self, inputs):
        self.inputs = [inputs] # Store activations for all layers
        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
            inputs = self.activation(inputs, self.activation_funcs[i])
            self.inputs.append(inputs)
        return inputs

    def mse_loss(self, O, E):
        return np.mean((O - E) ** 2)
    
    def binary_cross_entropy(self, O, E):
        return -np.sum(E * np.log(O) + (1 - E) * np.log(1 - O))
    
    def backward(self, O, E, loss_function="mse"):
        if loss_function == "mse":
            loss_grad = (2/O.size) * (O - E)
        elif loss_function == "bce":
            loss_grad = (O - E) / (O * (1 - O))
        for i in reversed(range(len(self.weights))):
            activation_derivative = self.activation(self.inputs[i+1], self.activation_funcs[i], derivative=True)
            delta = loss_grad * activation_derivative
            self.grad_weights[i] = np.dot(self.inputs[i].T, delta)
            self.grad_biases[i] = np.sum(delta, axis=0, keepdims=True)
            loss_grad = np.dot(delta, self.weights[i].T)
    
    def zero_grad(self, learning_rate=0.01):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.grad_weights[i]
            self.biases[i] -= learning_rate * self.grad_biases[i]

    def print_weights(self):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i}")
            print(w)
            print(b)

def main():
    # Model 1
    print("----- Model 1 -----")
    print("----- Task 1 -----")
    w1 = [
        np.array([[0.5, 0.6], [0.2, -0.6]]),
        np.array([[0.8], [-0.5]]),
        np.array([[0.6, -0.3]])
    ]
    b1 = [
        np.array([[0.3, 0.25]]),
        np.array([[0.6]]),
        np.array([[0.4, 0.75]])
    ]
    
    activation_funcs = ["relu", "linear", "linear"]
    nn1 = Network(w1, b1, activation_funcs)

    X = np.array([[1.5, 0.5]])
    E = np.array([[0.8, 1]])

    for _ in range(1000):
        O1 = nn1.forward(X)
        nn1.backward(O1, E)
        nn1.zero_grad(learning_rate=0.01)

    nn1.print_weights()
    loss_model1 = nn1.mse_loss(nn1.forward(X), E)
    print("----- Task 2 -----")
    print("Total Loss:", loss_model1)

    # Model 2
    print("----- Model 2 -----")
    print("----- Task 1 -----")
    w2 = [
        np.array([[0.5, 0.6], [0.2, -0.6]]),
        np.array([[0.8], [0.4]])
    ]
    b2 = [
        np.array([[0.3, 0.25]]),
        np.array([[-0.5]])
    ]
    activation_funcs = ["relu", "sigmoid"]
    nn2 = Network(w2, b2, activation_funcs)
    X = np.array([[0.75, 1.25]])
    E = np.array([[1]])
    for _ in range(1000):
        O2 = nn2.forward(X)
        nn2.backward(O2, E)
        nn2.zero_grad(learning_rate=0.01)

    nn2.print_weights()
    loss_model2 = nn2.mse_loss(nn2.forward(X), E)
    print("----- Task 2 -----")
    print("Total Loss:", loss_model2)


if __name__ == '__main__':
    main()