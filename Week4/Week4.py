import numpy as np

class Network:
    def __init__(self, weights, biases, activation_funcs):
        """
        Initialize the network
        :param weights: list of numpy arrays
        :param biases: list of numpy arrays
        :param activation_funcs: list of activation functions for each layer
        """
        self.weights = weights
        self.biases = biases
        self.activation_funcs = activation_funcs

    def activation(self, x, name):
        if name == "linear":
            return x
        elif name == "relu":
            return np.maximum(0, x)
        elif name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif name == "softmax":
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

    def forward(self, inputs):
        """
        Forward pass of the network
        :param inputs: list of input data
        :return: numpy array
        """
        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
            inputs = self.activation(inputs, self.activation_funcs[i])
        return inputs
    
    def mse_loss(self, O, E):
        """
        Calculate the mean squared error loss
        :param O: numpy array of outputs
        :param E: numpy array of expected outputs
        :return: float
        """
        return np.mean((O - E) ** 2)
    
    def binary_cross_entropy(self, O, E):
        """
        Calculate the binary cross-entropy loss
        :param O: numpy array of outputs
        :param E: numpy array of expected outputs
        :return: float
        """
        return -np.sum(E * np.log(O) + (1 - E) * np.log(1 - O))
    
    def categorical_cross_entropy(self, O, E):
        """
        Calculate the categorical cross-entropy loss
        :param O: numpy array of output probabilities
        :param E: numpy array of expected labels
        :return: float
        """
        return -np.sum(E * np.log(O))


def main():
    # Model 1
    w1 = [
        np.array([[0.5, 0.6],
                  [0.2, -0.6]]),

        np.array([[0.8, 0.4],
                  [-0.5, 0.5]])
    ]
    b1 = [
        np.array([0.3, 0.25]),
        np.array([0.6, -0.25])
    ]
    activation_m1 = ["relu", "linear"]
    nn1 = Network(w1, b1, activation_m1)
    inputs = [
        np.array([1.5, 0.5]),
        np.array([0, 1])       
    ]
    expected = [
        np.array([0.8, 1]),
        np.array([0.5, 0.5])
    ]
    print("------ Model 1 ------")
    for x, e in zip(inputs, expected):
        O = nn1.forward(x)
        loss = nn1.mse_loss(O, e)
        print(f"Outputs {list(O)}")
        print(f"Total Loss {loss}")
    
    # Model 2
    w2 = [
        np.array([[0.5, 0.6],
                  [0.2, -0.6]]),

        np.array([[0.8],
                  [0.4]])
    ]
    b2 = [
        np.array([0.3, 0.25]),
        np.array([-0.5])
    ]
    activation_m2 = ["relu", "sigmoid"]
    nn2 = Network(w2, b2, activation_m2)
    inputs = [
        np.array([0.75, 1.25]),
        np.array([-1, 0.5])       
    ]
    expected = [
        np.array([1]),
        np.array([0])
    ]
    print("------ Model 2 ------")
    for x, e in zip(inputs, expected):
        O = nn2.forward(x)
        loss = nn2.binary_cross_entropy(O, e)
        print(f"Outputs {list(O)}")
        print(f"Total Loss {loss}")
    
    # Model 3
    w3 = [
        np.array([[0.5, 0.6],
                  [0.2, -0.6]]),

        np.array([[0.8, 0.5, 0.3],
                  [-0.4, 0.4, 0.75]])
    ]
    b3 = [
        np.array([0.3, 0.25]),
        np.array([0.6, 0.5, -0.5])
    ]
    activation_m3 = ["relu", "sigmoid"]
    nn3 = Network(w3, b3, activation_m3)
    inputs_m3 = [
        np.array([1.5, 0.5]),
        np.array([0, 1])       
    ]
    expected_m3 = [
        np.array([1, 0, 1]),
        np.array([1, 1, 0])
    ]
    print("------ Model 3 ------")
    for x, e in zip(inputs_m3, expected_m3):
        O = nn3.forward(x)
        loss = nn3.binary_cross_entropy(O, e)
        print(f"Outputs {list(O)}")
        print(f"Total Loss {loss}")

    # Model 4
    w4 = [
        np.array([[0.5, 0.6],
                  [0.2, -0.6]]),

        np.array([[0.8, 0.5, 0.3],
                  [-0.4, 0.4, 0.75]])
    ]
    b4 = [
        np.array([0.3, 0.25]),
        np.array([0.6, 0.5, -0.5])
    ]
    activation_m4 = ["relu", "softmax"]
    nn4 = Network(w4, b4, activation_m4)
    inputs_m4 = [
        np.array([1.5, 0.5]),
        np.array([0, 1])       
    ]
    expected_m4 = [
        np.array([1, 0, 0]),
        np.array([0, 0, 1])
    ]
    print("------ Model 4 ------")
    for x, e in zip(inputs_m4, expected_m4):
        O = nn4.forward(x)
        loss = nn4.categorical_cross_entropy(O, e)
        print(f"Outputs {list(O)}")
        print(f"Total Loss {loss}")

if __name__ == "__main__":
    main()