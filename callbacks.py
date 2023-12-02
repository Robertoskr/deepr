import math


class Callback:
    def __init__(self, *args, **kwargs):
        pass

    def on_epoch_end(self, net, epoch: int, epoch_loss: float, epoch_val_loss: float):
        pass


class ExponentialDecay(Callback):
    def __init__(self, decay_rate: float, learning_rate: float | None = None):
        super().__init__()
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate

    def on_epoch_end(self, net, epoch: int, epoch_loss: float, epoch_val_loss: float):
        if not self.initial_learning_rate:
            self.initial_learning_rate = net.learning_rate

        epoch = epoch + 1

        net.optimizer.set_learning_rate(
            self.initial_learning_rate * math.exp(-self.decay_rate * epoch)
        )
