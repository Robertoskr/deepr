from .base import Layer 
import numpy as np 

class Dropout(Layer): 
    def __init__(self, p): 
        self.p = p 

    def _initialize_weights(self, X): 
        W = np.random.random(X.shape)
        W[W >= self.p] = 1
        W[W < self.p] = 0
        self.W = W 


    def forward(self, X, is_training: bool = False, *args, **kwargs): 
        if not is_training: 
            return X * (1 - self.p) 
        
        self._initialize_weights(X)
        return X * self.W 
    
    def backward(self, prev_grad, *args, **kwargs): 
        # During backpropagation, the gradient is only passed through the active neurons.
        return prev_grad * self.W