import numpy as np
from scipy.special import expit


def sigmoid(x):
    """
    Applies the sigmoid function element-wise.
    The Sigmoid function describes an S shaped curve.
    We cann pass the weighted sum of the inputs through this function to
    normalise them between 0 and 1.
    Check http://mathworld.wolfram.com/SigmoidFunction.html for details

    Numerically-stable sigmoid version, as seen here:
        http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function of a single value or numpy array of values
    :rtype: float or np.array
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


vectorized_sigmoid = np.vectorize(sigmoid)


def sigmoid_derivative(x):
    """
    Applies the sigmoid derivative function element-wise.
    This is the gradient of the Sigmoid curve.
    It indicates how confident we are about the existing weight.

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function's derivate of a single value or numpy array of values
    :rtype: float or np.array
    """
    result = sigmoid(x)
    return result * (1 - result)


vectorized_sigmoid_derivative = np.vectorize(sigmoid_derivative)


def perf_sigmoid(x):
    """
    Applies the sigmoid function element-wise using the high-perormance scipy implementation

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function of a single value or numpy array of values
    :rtype: float or np.array
    """
    return expit(x)


def perf_sigmoid_derivative(x):
    """
    Applies the sigmoid derivative function element-wise using the high-perormance scipy implementation

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function's derivate of a single value or numpy array of values
    :rtype: float or np.array
    """
    result = perf_sigmoid(x)
    return result * (1 - result)


def elliot(x):
    """
    Applies a fast approximation of sigmoid

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function of a single value or numpy array of values
    :rtype: float or np.array
    """
    s = 1  # steepness
    denominator = (1 + np.abs(x * s))
    return 0.5 * (x * s) / denominator + 0.5


def elliot_derivative(x):
    """
    Elliot's (fast sigmoid approximation) function derivative

    :param x: a number or numpy.array of numbers
    :return: the Sigmoid Function's derivate of a single value or numpy array of values
    :rtype: float or np.array
    """
    s = 1  # steepness
    denominator = (1 + np.abs(x * s))
    return 0.5 * s / denominator ** 2


def tanh(x):
    """
    Applies the tanh function element-wise.
    We cann pass the weighted sum of the inputs through this function to
    normalise them between 0 and 1.
    Check http://mathworld.wolfram.com/HyperbolicTangent.html for details

    :param x: a number or numpy.array of numbers
    :return: the tanh of a single value or numpy array of values
    :rtype: float or np.array
    """
    return np.tanh(x)


vectorized_tanh = np.vectorize(tanh)


def tanh_derivative(x):
    """
    Applies the tanh derivative function element-wise.

    :param x: a number or numpy.array of numbers
    :return: the tanh Function's derivate of a single value or numpy array of values
    :rtype: float or np.array
    """
    return 1 - np.power(tanh(x), 2)


vectorized_tanh_derivative = np.vectorize(tanh_derivative)


def main():
    matrix = np.array([[-0.59094028, 0.17897986, 0.88416026],
                       [-0.18231222, -0.30513586, -0.23577168],
                       [-0.10823045, -0.06333743, -0.03231125],
                       [0.27404187, 0.14167441, 0.37895093]])
    print("Sigmoid:")
    print(str(vectorized_sigmoid(matrix)))
    print("Performance Sigmoid:")
    print(str(perf_sigmoid(matrix)))
    print("Approximate Sigmoid:")
    print(str(elliot(matrix)))
    print("Sigmoid Derivative:")
    print(str(vectorized_sigmoid_derivative(matrix)))
    print("Performance Sigmoid Derivative:")
    print(str(perf_sigmoid_derivative(matrix)))
    print("Approximate Sigmoid Derivative:")
    print(str(elliot_derivative(matrix)))
    print("Tanh:")
    print(str(tanh(matrix)))
    print("Tanh Derivative:")
    print(str(tanh_derivative(matrix)))


if __name__ == "__main__":
    main()
