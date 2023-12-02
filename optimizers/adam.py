from .base import Optimizer
import numpy as np


class Adam(Optimizer):
    """
    Implementation of the Adam optimizer.
    """

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.momentums = []
        self.velocities = []
        self.t = 0

    def update_fn(self, grad, param):
        if len(self.momentums) < self.index + 1:
            self.momentums.append(np.zeros_like(param))
            self.velocities.append(np.zeros_like(param))

        self.t += 1
        self.momentums[self.index] = (
            self.beta1 * self.momentums[self.index] + (1 - self.beta1) * grad
        )

        self.velocities[self.index] = self.beta2 * self.velocities[self.index] + (
            1 - self.beta2
        ) * (grad * grad)

        learning_rate = (
            self.learning_rate
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )

        param -= (
            learning_rate
            * self.momentums[self.index]
            / (np.sqrt(self.velocities[self.index]) + self.epsilon)
        )
