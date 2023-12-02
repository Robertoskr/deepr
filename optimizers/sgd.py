from .base import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = []

    def update_fn(self, grad, param):
        if len(self.velocities) < self.index + 1:
            self.velocities.append(np.zeros_like(grad))

        self.velocities[self.index] = (
            self.momentum * self.velocities[self.index] - self.learning_rate * grad
        )

        param += self.velocities[self.index]
