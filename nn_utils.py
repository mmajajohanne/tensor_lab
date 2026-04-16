import numpy as np
from openml.datasets import get_dataset


def _split_data_in_train_val(x_data, y_data, n_val=10000, seed=57):
    rng = np.random.default_rng(seed=seed)
    n = x_data.shape[0]
    random_indices = rng.permutation(n)

    n_train = len(x_data) - n_val
    train_indices = random_indices[:n_train]
    val_indices = random_indices[n_train:]

    x_train = x_data[train_indices]
    x_val = x_data[val_indices]

    y_train = y_data[train_indices]
    y_val = y_data[val_indices]

    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def load_mnist_data(scale_x_data=True):
    """Laster MNIST-datasettet og deler det opp i trenings-, validerings- og testsett."""
    mnist = get_dataset(554)
    x_data, y_data, _, _ = mnist.get_data(dataset_format="dataframe", target="class")
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    x_main = x_data[:60000]
    y_main = y_data[:60000].astype(int)
    x_test = x_data[60000:]
    y_test = y_data[60000:].astype(int)

    all_data = _split_data_in_train_val(x_data=x_main, y_data=y_main)
    all_data["x_test"] = x_test
    all_data["y_test"] = y_test.astype(int)

    if scale_x_data is True:
        all_data["x_train"] = all_data["x_train"].astype(np.float32) / 255
        all_data["x_val"] = all_data["x_val"].astype(np.float32) / 255
        all_data["x_test"] = all_data["x_test"].astype(np.float32) / 255

    return all_data


def integer_one_hot_encode(x_array, max_int=None):
    """Gjør om en array av heltall til one-hot-kodet form."""
    if max_int is None:
        max_int = x_array.max()
    one_hot_array = np.zeros((x_array.shape[0], max_int + 1))
    one_hot_array[np.arange(x_array.shape[0]), x_array] = 1
    return one_hot_array.astype(np.uint8)


def softmax(x_data):
    """Regner ut softmax på en numerisk stabil måte ved å trekke fra maks-verdien."""
    shifted_x = x_data - np.max(x_data, axis=1, keepdims=True)
    return np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=1, keepdims=True)


def calculate_multiclass_cross_entropy(targets, predictions):
    """
    Regner ut multiklasse kryssentropi (log loss).

    Argumenter:
    - targets    : [n]-array med sanne klasser som heltall.
    - predictions: [n x c]-array med logit-verdier (ikke sannsynligheter).

    Returnerer:
    - float: Gjennomsnittlig kryssentropi.
    """
    probs = softmax(predictions)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    n = len(targets)
    return -np.mean(np.log(probs[np.arange(n), targets]))


def calculate_accuracy(targets, predictions):
    """
    Regner ut nøyaktighet.

    Argumenter:
    - targets    : [n]-array med sanne klasser.
    - predictions: [n]-array med predikerte klasser.

    Returnerer:
    - float: Andel korrekte prediksjoner.
    """
    return np.mean(targets == predictions)


class IdentityActivation:
    def __call__(self, x_data):
        return x_data

    def diff(self, x_data):
        return np.ones(x_data.shape)


class NeuralNetwork:
    """
    Fullt koblet feed-forward nevralt nettverk trent med SGD og tilbakepropagering.

    Vektmatriser har neste lag som rader og forrige lag som kolonner: [n(l) x n(l-1)].
    """

    def __init__(self, layer_sizes, activation_functions, initialization_method="normal", seed=57):
        np.random.seed(seed=seed)
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self._initialize_weights(layer_sizes=layer_sizes, initialization_method=initialization_method)

    def _initialize_weights(self, layer_sizes, initialization_method):
        self.biases = [np.zeros((1, layer_sizes[i])) for i in range(1, len(layer_sizes))]

        method = initialization_method.lower().strip()
        self.weights = []

        if method == "zeros":
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.zeros((layer_sizes[i + 1], layer_sizes[i])))
        elif method == "ones":
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.ones((layer_sizes[i + 1], layer_sizes[i])))
        elif method == "normal":
            for i in range(len(layer_sizes) - 1):
                std = 1 / np.sqrt(layer_sizes[i])
                self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * std)
        else:
            raise ValueError(f'Argument `method` must be in ["zeros", "ones", "normal"]. Was {method}.')

    def forward(self, x_data):
        """
        Sender data fremover gjennom nettverket.

        Argumenter:
        - x_data: [n x m]-array med inputdata.

        Returnerer:
        - [n x c]-array med logit-verdier fra siste lag.
        """
        self.activations = [x_data]
        self.weighted_sums = []

        current = x_data
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i].T + self.biases[i]
            self.weighted_sums.append(z)
            current = self.activation_functions[i](z)
            self.activations.append(current)

        return current

    def predict(self, x_data):
        """
        Predikerer klasser for inputdata.

        Argumenter:
        - x_data: [n x p]-array med inputdata.

        Returnerer:
        - [n]-array med predikerte klasser.
        """
        logits = self.forward(x_data)
        return np.argmax(logits, axis=1)

    def count_parameters(self):
        """Teller totalt antall parametere (vekter + bias) i nettverket."""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total

    def _backprop(self, preds, targets):
        deltas = []
        delta_L = preds - targets
        deltas.append(delta_L)
        for i in range(1, self.n_layers - 1):
            index = self.n_layers - i - 1
            prev_delta = deltas[0]
            delta = prev_delta @ self.weights[index]
            delta = delta * self.activation_functions[-(i + 1)].diff(self.weighted_sums[index - 1])
            deltas.insert(0, delta)
        return deltas

    def _sgd(self, deltas, eta, n_data):
        for i in range(self.n_layers - 1):
            d_biases = deltas[i].sum(axis=0)
            self.biases[i] -= (eta / n_data) * d_biases

            d_weights = deltas[i].T @ self.activations[i]
            self.weights[i] -= (eta / n_data) * d_weights

    def _run_single_epoch(self, x_train, y_train, minibatch_size, eta):
        b = minibatch_size
        n = x_train.shape[0]
        for i in range(int(n / b)):
            upper_index = np.min((n, (i + 1) * b))
            n_data = upper_index - i * b
            batch = x_train[i * b : upper_index]
            targets = y_train[i * b : upper_index]
            preds = self.forward(batch)
            deltas = self._backprop(preds, targets)
            self._sgd(deltas, eta, n_data)

    def _perform_evaluation(self, x_train, y_train, n_epoch, loss_func, accuracy_func, eval_set=None):
        train_logits = self.forward(x_train)
        self.train_losses[n_epoch] = loss_func(y_train, train_logits)
        train_preds = self.predict(x_train)
        self.train_accuracies[n_epoch] = accuracy_func(y_train, train_preds)

        if eval_set is not None:
            x_val, y_val = eval_set
            val_logits = self.forward(x_val)
            self.val_losses[n_epoch] = loss_func(y_val, val_logits)
            val_preds = self.predict(x_val)
            self.val_accuracies[n_epoch] = accuracy_func(y_val, val_preds)

            print(f"Train-loss: {self.train_losses[n_epoch]:.5f}, ", end="")
            print(f"Validerings-loss: {self.val_losses[n_epoch]:.5f}. ", end="")
            print(f"Train-nøyaktighet: {self.train_accuracies[n_epoch]:.5f}, ", end="")
            print(f"Validerings-nøyaktighet: {self.val_accuracies[n_epoch]:.5f}")

    def train(self, x_train, y_train, eta, n_epochs, loss_func, accuracy_func, minibatch_size=64, eval_set=None):
        self.train_losses = np.zeros(n_epochs)
        self.train_accuracies = np.zeros(n_epochs)
        if eval_set is not None:
            self.val_losses = np.zeros(n_epochs)
            self.val_accuracies = np.zeros(n_epochs)

        y_train_one_hot = integer_one_hot_encode(y_train, max_int=9)
        for n_epoch in range(n_epochs):
            print(f"Epoke [{n_epoch + 1} / {n_epochs}]")
            self._run_single_epoch(x_train=x_train, y_train=y_train_one_hot, minibatch_size=minibatch_size, eta=eta)
            self._perform_evaluation(
                x_train=x_train,
                y_train=y_train,
                loss_func=loss_func,
                accuracy_func=accuracy_func,
                n_epoch=n_epoch,
                eval_set=eval_set,
            )
