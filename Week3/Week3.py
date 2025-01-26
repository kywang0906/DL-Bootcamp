import numpy as np

class Network:
    def __init__(self, weights, biases):
        """
        Initialize the network
        :param weights: list of numpy arrays where each array represents the weights
        :param biases: list of numpy arrays where each array represents the biases
        """
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        """
        Forward pass of the network
        :param inputs: list of input data
        :return: numpy array
        """
        for w, b in zip(self.weights, self.biases):
            inputs = np.dot(inputs, w) + b
        return inputs


def main():
    w1 = [
        np.array([[0.5, 0.6], [0.2, -0.6]]), 
        np.array([[0.8], [0.4]])
    ]
    b1 = [
        np.array([0.3, 0.25]), 
        np.array([-0.5])
    ]
    w2 = [
        np.array([[0.5, 0.6], [1.5, -0.8]]),
        np.array([[0.6], [-0.8]]),
        np.array([[0.5, -0.4]])
    ]
    b2 = [
        np.array([0.3, 1.25]),
        np.array([0.3]),
        np.array([0.2, 0.5])
    ]
    network1 = Network(w1, b1)
    network2 = Network(w2, b2)
    print("----- Model 1 -----")
    print(network1.forward([1.5, 0.5]))
    print(network1.forward([0, 1]))
    print("----- Model 2 -----")
    print(network2.forward([0.75, 1.25]))
    print(network2.forward([-1, 0.5])) 

if __name__ == '__main__':
    main()