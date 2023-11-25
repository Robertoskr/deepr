"""
Base Neural network class. 
A neural network is a combination of layers. Which work together to learn and make predictions. 
"""
from .optimizers import SGD


class NeuralNetwork:
    def __init__(self, *layers, **kwargs):
        self.layers = list(layers)
        self.n_layers = len(self.layers)
        self.is_training = False
        self.X = None
        self.X_test = None

    def get_layer_params(self, param_key):
        for layer in self.layers:
            if hasattr(layer, param_key):
                for layer_param_key_value in getattr(layer, param_key):
                    yield getattr(layer, layer_param_key_value)

    @property
    def params(self):
        yield from self.get_layer_params("params")

    @property
    def grads(self):
        yield from self.get_layer_params("grads")

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

    def backward(self, X, y):
        a = self.predict(X)
        grad = self.loss_fn.derivative(a, y)
        for i, layer in enumerate(reversed(self.layers)):
            is_first_layer = i == self.n_layers - 1
            grad = layer.backward(grad, is_first_layer)

    def optimize(self, X, y):
        self.backward(X, y)
        self.optimizer.step(self.params, self.grads)

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

        percent_epoch_completed = (epoch_idx + 1) / total_epoch_iters
        print_str = f"Epoch {epoch + 1}/{self.epochs}, loss: {loss:.4f}, completed: {percent_epoch_completed:.3f}"
        print_str += f", learning rate: {self.optimizer.learning_rate}"
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

    def _get_batch(self, X, batch_idx, batch_size):
        l, r = batch_idx * batch_size, (batch_idx + 1) * batch_size
        n_dims = len(X.shape) - 1
        if n_dims == 1:
            return X[:, l:r]
        elif n_dims == 2:
            return X[:, :, l:r]
        elif n_dims == 3:
            return X[:, :, :, l:r]

    def _update_loss(self, loss, running_loss, epoch_idx):
        running_loss += loss
        epoch_loss = running_loss / (epoch_idx + 1)
        return running_loss, epoch_loss

    def fit(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        batch_size=36,
        epochs=10,
        optimizer=SGD(learning_rate=0.001),
        shuffle: bool = False,
        n_epochs_to_log=None,
        callbacks: list | None = None,
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

        if n_epochs_to_log is None:
            n_epochs_to_log = epochs
        self.n_epochs_to_log = n_epochs_to_log
        self.loss_fn = loss
        self.epochs = epochs
        self.callbacks = callbacks or []
        self.optimizer = optimizer

        self.set_training_status(True)
        self._fit(X_transposed, y_transposed, batch_size)
        self.set_training_status(False)

    def _fit(
        self,
        X,
        y,
        batch_size=36,
    ):
        self.X = X
        self.y = y

        # iterate for every epoch.
        n_samples = X.shape[-1]
        n_batchs_per_epoch = n_samples // batch_size
        if n_samples > n_batchs_per_epoch * batch_size:
            n_batchs_per_epoch += 1

        for epoch_idx in range(self.epochs):
            running_loss = 0.0
            running_val_loss = 0.0
            epoch_loss = 0.0
            epoch_val_loss = 0.0

            should_print_progress = self.should_print_progress(epoch_idx)

            for batch_idx in range(n_batchs_per_epoch):
                X_batch = self._get_batch(X, batch_idx, batch_size)
                y_batch = self._get_batch(y, batch_idx, batch_size)

                self.optimize(X_batch, y_batch)
                loss = self.loss(X_batch, y_batch)
                running_loss, epoch_loss = self._update_loss(
                    loss, running_loss, batch_idx
                )
                if self.X_test is not None:
                    val_loss = self.loss(self.X_test, self.y_test)
                    running_val_loss, epoch_val_loss = self._update_loss(
                        val_loss, running_val_loss, batch_idx
                    )

                if should_print_progress:
                    self.print_progress(
                        epoch_idx,
                        batch_idx,
                        n_batchs_per_epoch - 1,
                        epoch_loss,
                        val_loss=epoch_val_loss,
                    )
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch_idx, epoch_loss, epoch_val_loss)
