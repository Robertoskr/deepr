import numpy as np


class DerivableFunction:
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
class NoActivation(DerivableFunction):
    def base(self, X):
        return X

    def derivative(self, X):
        return 1


class Relu(DerivableFunction):
    def base(self, X):
        X = X.copy()
        X[X < 0] = 0
        return X

    def derivative(self, X):
        X = X.copy()
        X[X <= 0] = 0
        X[X > 0] = 1
        return X


class Softmax(DerivableFunction):
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


#
# -- LOSS FUNCTIONS
#
class MSE(DerivableFunction):
    def base(self, a, y):
        return np.mean((a - y) ** 2)

    def derivative(self, a, y):
        # output is d * n
        return (2 * (a - y)) / a.shape[1]


class CrossEntropy(DerivableFunction):
    def base(self, a, y):
        # Small constant for numerical stability
        epsilon = 1e-12
        a = np.clip(a, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(a)) / y.shape[1]

    def derivative(self, a, y):
        # For one-hot encoded labels, derivative simplifies to:
        return a - y
