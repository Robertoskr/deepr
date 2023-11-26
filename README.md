# deepr

Deep learning framework

Deepr is a deep learning framework created for learning purposes along with the deep learning course in my personal website.

## Supported:

### Layers:

- Fully Connected layer.
- Dropout layer.
- Convolutional layer.
- Flatten layer.
- Max pooling layer.

### Loss functions:

- Mean Squared error.
- Cross entropy.

### Activation functions:

- Relu.
- Softmax.
- Sigmoid.

### Optimizers:

- SGD
- SGDM (SGD with momentum.)
- RMSProps (In progress)
- Adam (In progress)

### Regularization functions:

- L1
- L2

## Example usage:

```python
from keras.datasets import mnist
import numpy as np
from deepr.layers import Convolutional, Dropout, Dense, Flatten, MaxPooling
from deepr.net import NeuralNetwork
from deepr.callbacks import ExponentialDecay
from deepr.functions import MSE, Relu, Softmax, CrossEntropy, Sigmoid
from deepr.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder

data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

y_train = y_train.toarray()
y_test = y_test.toarray()

net = NeuralNetwork(
    Convolutional((1, 28, 28), 4, 1, random_kernels=True),
    # You can do it like this
    Sigmoid(),
    MaxPooling(2, stride=2),
    Flatten(),
    # Or you can define the activation function here.
    Dense(1 * ((23 // 2) + 1) ** 2, 10, Softmax())
)

optimizer = SGD(learning_rate=0.001)

net.fit(
    X_train,
    y_train,
    optimizer=optimizer,
    batch_size=72,
    epochs=10,
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
