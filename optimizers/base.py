class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.number_of_params = 0
        self.index = 0
        self.epoch = 0

    def update_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update_epoch(self, epoch: int):
        self.epoch = epoch

    def update_index(self):
        """
        Usefull index for optimizers that store some state for each parameter
        the index corresponds to the parameter index.
        """
        self.index += 1
        if self.index >= self.number_of_params:
            self.index = 0

    def step(self, params, grads):
        params = list(params)
        grads = list(grads)

        self.number_of_params = len(params)

        for param, grad in zip(params, grads):
            self.update_fn(grad, param)
            self.update_index()

    def update_fn(self, grad, param):
        raise NotImplementedError
