import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.weights = []
        self.biases = []

        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.normal(
                0, 1, (layer_sizes[i], layer_sizes[i - 1])))
            self.biases.append(np.zeros((layer_sizes[i], 1)))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return self.sigmoid(x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        A = None
        for i in range(len(self.weights)):
            if i == 0:
                A = self.activation(np.dot(self.weights[i], x) + self.biases[i])
            else:
                A = self.activation(np.dot(self.weights[i], A) + self.biases[i])
        return A

    # Definition of some activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        return np.maximum(0.01 * x, x)

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def linear(self, x):
        return x
