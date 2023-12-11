import numpy as np
from .layers.base import Layer


class DerivableFunction(Layer):
    def __init__(self, *args, **kwargs):
        pass

    def base(self, *args, **kwargs):
        raise NotImplementedError()

    def derivative(self, *args, **kwargs):
        raise NotImplementedError()


#
# -- REGULARIZATION FUNCTIONS
#
class RegularizationFunction(DerivableFunction):
    def __init__(self, rate=0.01):
        self.rate = rate

    def base(self, W, *args, **kwargs):
        return W


class L1(RegularizationFunction):
    def base(self, W):
        return self.rate * np.sum(np.abs(W))  # L1 norm of W.

    def derivative(self, W):
        return self.rate * np.sign(W)  # Derivative of L1 norm is the sign of W.


class L2(RegularizationFunction):
    def base(self, W):
        return self.rate * np.sum(W**2)  # L2 norm squared.

    def derivative(self, W):
        return 2 * self.rate * W  # Derivative of L2 norm squared is 2*W.


#
# -- ACTIVATION FUNCTIONS
#
class Activation(DerivableFunction, Layer):
    # These functions are to support the layer functionality, only
    # usable for activation functions.
    # Thanks to this, you can do:
    # NeuralNetwork(Dense(2, 5), Softmax())
    # insead of:
    # NeuralNetwork(Dense(2, 5), activation=Softmax())
    def forward(self, X, *args, **kwargs):
        return self.base(X)

    def backward(self, prev_grad, *args, **kwargs):
        return prev_grad * self.derivative(prev_grad)


class NoActivation(Activation):
    def base(self, X):
        return X

    def derivative(self, X):
        return 1


class Relu(Activation):
    def base(self, X):
        return np.maximum(0, X)

    def derivative(self, X):
        return np.where(X > 0, 1, 0)


class LeakyRelu(Activation):
    def __init__(self, alpha=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def base(self, X):
        return np.maximum(self.alpha * X, X)

    def derivative(self, X):
        return np.where(X > 0, 1, self.alpha)


class Softmax(Activation):
    def base(self, X):
        # stable softmax implementation
        # Apply along the correct axis and keepdims for proper broadcasting
        exps = np.exp(X - np.max(X, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def derivative(self, X):
        # due to the complexity of this derivative.
        # and the cardinality, we can only use the softmax activation with the
        # cross entropy loss function.
        # the 1 will be multiplied by the cross entropy loss function and we will keep it.
        return 1


class Sigmoid(Activation):
    def base(self, X):
        # Clip X to prevent overflow and underflow in exp(-X)
        X_clipped = np.clip(X, -500, 500)
        return 1 / (1 + np.exp(-X_clipped))

    def derivative(self, X):
        sigmoid = self.base(X)
        return sigmoid * (1 - sigmoid)


class Tanh(Activation):
    def base(self, X):
        return np.tanh(X)

    def derivative(self, X):
        return 1 - np.tanh(X) ** 2


#
# -- LOSS FUNCTIONS
#
class MSE(DerivableFunction):
    def base(self, a, y):
        return np.mean((a - y) ** 2)

    def derivative(self, a, y):
        # output is d * n
        return -1 * ((2 * (a - y)) / a.shape[1])


class CrossEntropy(DerivableFunction):
    def base(self, a, y):
        # Small constant for numerical stability
        epsilon = 1e-12
        a = np.clip(a, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(a)) / y.shape[1]

    def derivative(self, a, y):
        # For one-hot encoded labels, derivative simplifies to:
        return a - y
