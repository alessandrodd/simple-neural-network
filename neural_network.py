import logging
import math

import fastdot
import numpy as np
from activation_functions import elliot, symmetric_elliot, symmetric_elliot_derivative, relu_derivative, relu, \
    leaky_relu, leaky_relu_derivative
from activation_functions import elliot_derivative
from activation_functions import perf_sigmoid
from activation_functions import perf_sigmoid_derivative
from activation_functions import tanh
from activation_functions import tanh_derivative


class NeuronLayer(object):
    """
    Represents a single Neural Network layer. Implements any input layer, output layer or hidden layer
    """

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron, activation_function_name='sigmoid',
                 biased=False):
        """

        :param number_of_neurons: neurons in the layer
        :param number_of_inputs_per_neuron: inputs to this layer; should be equal to the number of
                                           neurons in the previous layer or, if this is the first layer
                                           in a Neural Network, should be equal to the input dimensionality
        :param activation_function_name: a string that indicates which activation should be used, e.g. 'tanh'
        :param biased: True if it has a bias
        """
        # We use Xavier initialization. Here for more information:
        # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        variance = 1 / number_of_inputs_per_neuron
        self.weights = np.random.normal(0, math.sqrt(variance), size=(number_of_inputs_per_neuron, number_of_neurons))
        # Choose an activation function for forward propagation
        # and the corrispettive derivative for backward propagation
        if activation_function_name == 'sigmoid':
            self.activation_function = perf_sigmoid
            self.activation_derivative = perf_sigmoid_derivative
        elif activation_function_name == 'tanh':
            self.activation_function = tanh
            self.activation_derivative = tanh_derivative
        elif activation_function_name == 'elliot':
            self.activation_function = elliot
            self.activation_derivative = elliot_derivative
        elif activation_function_name == 'symmetric_elliot':
            self.activation_function = symmetric_elliot
            self.activation_derivative = symmetric_elliot_derivative
        elif activation_function_name == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        elif activation_function_name == 'leaky_relu':
            self.activation_function = leaky_relu
            self.activation_derivative = leaky_relu_derivative
        else:
            logging.error("Invalid activation function: '{0}'".format(activation_function_name))
            return
        if biased:
            self.bias = np.zeros((1, number_of_neurons))
        else:
            self.bias = None


class NeuralNetwork(object):
    """
    A simple Neural Network implementation
    """

    def __init__(self, layers, learning_rate=0.1, step_decay_factor=0.5, epochs_drop=10):
        """

        :param layers: a list of Neuron Layers
        :param learning_rate: training parameter that controls the size of weight and bias changes in learning of
                               the training algorithm.
        """
        if len(layers) < 1:
            logging.error("Invalid layers size: '{0}' (min is 1)".format(len(layers)))
            return
        self.layers = layers
        if learning_rate > 1 or learning_rate <= 0:
            logging.error("Invalid learning_rate: '{0}' (min is 0, max is 1)".format(learning_rate))
            return
        else:
            self.learning_rate = learning_rate

        assert step_decay_factor >= 0
        self.step_decay_factor = step_decay_factor
        assert epochs_drop > 0
        self.epochs_drop = epochs_drop

    def train(self, input_values, output_values, epochs):
        """
        Trains the neural network

        :param input_values: an n*m matrix of input data where each row is an entry (n entries) and m corresponds to
                             the input dimensionality
        :param output_values: an n*o matrix with the expected results
        :param epochs: number of training iterations
        """
        if epochs < 1:
            logging.error("Invalid epochs : '{0}' (min is 1)".format(epochs))
            return
        elif output_values.shape[0] != input_values.shape[0]:
            logging.error("Output dimension ({0}) doesn't match input dimension ({1})".format(output_values.shape[0],
                                                                                              input_values.shape[0]))
            return
        logging.debug("Training started")
        learning_rate = self.learning_rate
        for iteration in range(epochs):
            progress = iteration / epochs * 100
            if progress.is_integer():
                logging.debug("{0}% complete".format(progress))

            # learning rate decay
            if (iteration + 1) % self.epochs_drop == 0:
                learning_rate = learning_rate * self.step_decay_factor

            # Forward Propogation
            outputs = self.compute_outputs(input_values)

            # Backpropagation
            # Propagate the error from the rightmost layer (output layer) back to the first
            # leftmost layer
            next_layer_delta = None
            for j in range(len(self.layers) - 1, -1, -1):
                # for the latest layer, the error is calculated from the expected output
                if j == (len(self.layers) - 1):
                    error = outputs[j] - output_values
                    error = np.power(error, 2) * np.sign(error)
                else:
                    error = fastdot.dot(next_layer_delta, self.layers[j + 1].weights.T)
                slope = self.layers[j].activation_derivative(outputs[j])
                # Hadamard product, i.e. entrywise product
                delta = np.multiply(error, slope)
                next_layer_delta = delta
                # Calculate how much to adjust the weights by
                if j == 0:
                    adjustment = fastdot.dot(input_values.T, delta) / input_values.shape[0]
                else:
                    adjustment = fastdot.dot(outputs[j - 1].T, delta) / input_values.shape[0]
                self.layers[j].weights += np.multiply(-adjustment, learning_rate)
                if self.layers[j].bias is not None:
                    # bias are like neurons with fixed input 1; we sum on columns because it's like multiplying
                    # a one-row matrix with all ones
                    self.layers[j].bias += np.multiply(-np.sum(delta, axis=0, keepdims=True), learning_rate)

            print(self.calculate_loss(input_values, output_values))

    def compute_outputs(self, inputs):
        """
        Evaluate a set of inputs through the Neural Network

        :param inputs: a numpy matrix where each row is an input value of dimension equals to the number of columns
        :return: a list of numpy arrays where the latest corresponds to the output of the Neural Network
        """
        outputs = []
        for i in range(len(self.layers)):
            s_i = fastdot.dot(inputs, self.layers[i].weights)
            if self.layers[i].bias is not None:
                s_i += self.layers[i].bias
            z_i = self.layers[i].activation_function(s_i)
            outputs.append(z_i)
            # the input to the next layer is the output of this layer
            inputs = z_i
        return outputs

    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self, inputs, outputs_values):
        prediction = self.evaluate(inputs)
        # Calculating the loss
        logloss = outputs_values * np.log(prediction)
        logloss += (1 - outputs_values) * np.log(1 - prediction)
        data_loss = np.sum(logloss)
        return 1. / len(prediction) * data_loss

    def evaluate(self, inputs):
        """
        Evaluates one or more inputs

        :param inputs: a numpy matrix where each row is an input value of dimension equals to the number of columns
        :return: a numpy matrix where each row is the computed output
        :rtype: np.array
        """
        return self.compute_outputs(inputs)[-1]

    def classify(self, inputs):
        output = self.evaluate(inputs)
        for i in range(len(output)):
            output[i][0] = round(output[i][0], 0)
        return output

    def score(self, inputs, expected_outputs):
        output = self.classify(inputs)
        score = 0
        for i in range(len(output)):
            if output[i][0] == expected_outputs[i][0]:
                score += 1
        score = score / len(output)
        return score

    def print_weights(self):
        """
        The neural network prints its weights
        """
        for i in range(len(self.layers)):
            print("Layer {0}:".format(i))
            print(str(self.layers[i].weights))


def main():
    logging.basicConfig(level=logging.INFO)
    # Layer 1: 5 neurons, each with 4 inputs
    layer1 = NeuronLayer(5, 3, 'leaky_relu', True)
    # Layer 2: 5 neurons, each with 5 inputs
    layer2 = NeuronLayer(5, 5, 'leaky_relu', True)
    # Layer 3: 1 neuron with 5 inputs (output layer)
    layer3 = NeuronLayer(1, 5, 'sigmoid', True)

    # Combine the layers to create a neural network
    layers = [layer1, layer2, layer3]
    neural_network = NeuralNetwork(layers, learning_rate=0.1)

    print("Random initial weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    # training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
    training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print("Layers weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Test case with a new situation: [1, 1, 0], [0, 0, 1], [0, 1, 1] -> ?: (should be [[0],[0],[1]])")
    output = neural_network.evaluate(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]]))
    print(output)


if __name__ == "__main__":
    main()
