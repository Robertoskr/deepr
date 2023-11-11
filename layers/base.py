class Layer:
    """
    Base layer class, a neural network is made of a list layers.
    One on on 'top' of the other.
    """

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, X, is_training: bool = False):
        raise NotImplementedError()

    def backward(self, prev_grad, learning_rate):
        raise NotImplementedError()
