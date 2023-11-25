class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8

    def update_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            self.update_fn(grad, param)

    def update_fn(self, grad, param):
        raise NotImplementedError


class SGD(Optimizer):
    def update_fn(self, grad, param):
        param -= self.learning_rate * grad
