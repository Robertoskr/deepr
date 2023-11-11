# deepr
Deep learning framework

Deepr is a deep learning framework created for learning purposes along with the deep learning course in my personal website. 

## Supported: 
### Layers: 
- Fully Connected layer.
- Dropout layer.
- Convolutional layer 
- Pooling layers (coming soon)
### Loss functions: 
- Mean Squared error.
- Cross entropy.
### Activation functions: 
- Relu.
- Softmax.
- Sigmoid. 
### Regularization functions: 
- L1
- L2

## Example usage: 
```python
from keras.datasets import mnist 
import numpy as np 
from deepr.layers import Convolutional, Dense, Flatten
from deepr.net import NeuralNetwork
from deepr.functions import MSE, Relu, Softmax, CrossEntropy, Sigmoid
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data 
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

y_train = y_train.toarray()
y_test = y_test.toarray()

net = NeuralNetwork(
    Convolutional((1, 28, 28), 6, 3), 
    # You can do it like this 
    Sigmoid(), 
    Flatten(), 
    # Or you can define the activation function here. 
    Dense(3 * 23 * 23, 10, Softmax()), 
)

net.fit(
    X_train[0:1000], 
    y_train[0:1000], 
    learning_rate=0.0005, 
    batch_size=72, 
    epochs=250, 
    n_epochs_to_log=100,
    loss=CrossEntropy()
)

def predict(index): 
    image, target = X_train[index:index+1], y_train[index]

    pred = net.predict(image)[0]
    pred_max = pred.max()
    pred[pred >= pred_max] = 1
    pred[pred < pred_max] = 0
    print(pred, target)
    correct = np.sum(pred == target) >= 1
    print(f"Correct: {correct}")

predict(20)
```
