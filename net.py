"""
Base Neural network class. 
A neural network is a combination of layers. Which work together to learn and make predictions. 
"""


class NeuralNetwork:
    def __init__(self, *layers, **kwargs):
        self.layers = list(layers)
        self.n_layers = len(self.layers)
        self.is_training = False

    def predict(self, X):
        if not self.is_training:
            X = self._transform_input(X)

        a = X
        for layer in self.layers:
            a = layer.forward(a, is_training=self.is_training)

        if not self.is_training:
            a = self._transform_input(a)
        return a

    def set_training_status(self, is_training: bool):
        self.is_training = is_training

    def optimize(self, X, y):
        a = self.predict(X)
        grad = self.loss_fn.derivative(a, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)

    def should_print_progress(self, epoch):
        print_multiply = self.epochs // self.n_epochs_to_log
        # Print at the start of the first epoch and at the end of the last epoch
        if epoch == 0 or epoch == self.epochs - 1:
            return True

        # Print if it's the last iteration of an epoch according to the logging frequency
        if epoch % print_multiply == 0:
            return True
        return False

    def print_progress(self, epoch, epoch_idx, total_epoch_iters, loss, val_loss=None):
        end = "\r"
        if epoch_idx == total_epoch_iters:
            end = "\r\n"

        print_str = f"Epoch {epoch + 1}/{self.epochs}, loss: {loss:.4f}"
        if val_loss is not None:
            print_str += f", val loss: {val_loss}"

        print(print_str, end=end)

    def loss(self, X, y):
        a = self.predict(X)
        loss = self.loss_fn.base(a, y)
        return loss

    def _transform_input(self, X, y=None):
        X = X.T
        if y is not None:
            return X, y.T
        return X

    def fit(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        batch_size=36,
        epochs=10,
        learning_rate=0.001,
        shuffle: bool = False,
        n_epochs_to_log=10,
        loss=None,
    ):
        # first thing transpose X. We work with columns as indivisual entries.
        X_transposed, y_transposed = self._transform_input(X, y)
        if X_test is not None:
            X_test_transposed, y_test_transposed = self._transform_input(X_test, y_test)
            self.X_test = X_test_transposed
            self.y_test = y_test_transposed

        if shuffle:
            pass

        self.learning_rate = learning_rate
        self.n_epochs_to_log = n_epochs_to_log
        self.loss_fn = loss
        self.epochs = epochs

        self.set_training_status(True)
        self._fit(X_transposed, y_transposed, batch_size, epochs, learning_rate)
        self.set_training_status(False)

    def _fit(
        self,
        X,
        y,
        batch_size=36,
        epochs=10,
        learning_rate=0.001,
    ):
        self.X = X
        self.y = y

        # iterate for every epoch.
        data_dims, n_samples = X.shape
        n_batchs_per_epoch = n_samples // batch_size
        if n_samples > n_batchs_per_epoch * batch_size:
            n_batchs_per_epoch += 1

        for epoch_idx in range(epochs):
            for batch_idx in range(n_batchs_per_epoch):
                l, r = batch_idx * batch_size, (batch_idx + 1) * batch_size
                X_batch = X[:, l:r]  # Selecting all features for the batch samples
                y_batch = y[:, l:r]

                self.optimize(X_batch, y_batch)
                if self.should_print_progress(epoch_idx):
                    loss = self.loss(self.X, self.y)
                    val_loss = None
                    if self.X_test is not None:
                        val_loss = self.loss(self.X_test, self.y_test)
                    self.print_progress(
                        epoch_idx,
                        batch_idx,
                        n_batchs_per_epoch - 1,
                        loss,
                        val_loss=val_loss,
                    )
