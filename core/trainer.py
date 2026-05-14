import numpy as np

from core.logistics_regression import LogisticRegressionManual


def _train_test_split(X, y, train_ratio=0.8, shuffle=True, random_state=42):
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    train_size = int(round(n_samples * train_ratio))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def _iterate_minibatches(X, y, batch_size=32, shuffle=True, random_state=42):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _binary_cross_entropy(y_true, y_prob):
    epsilon = 1e-9
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    return float(
        -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    )


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def train_model_process(
    X,
    y,
    *,
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
    train_ratio=0.8,
    shuffle=True,
    random_state=42,
):
    """
    Train Logistic Regression (manual) with mini-batch Gradient Descent.

    Returns:
        model: LogisticRegressionManual (trained weights/bias)
        history: dict with per-epoch metrics for train/val
        split: dict with sizes
        y_val, y_val_pred: for evaluation downstream
    """

    X_train, X_val, y_train, y_val = _train_test_split(
        X, y, train_ratio=train_ratio, shuffle=shuffle, random_state=random_state
    )

    model = LogisticRegressionManual(learning_rate=learning_rate, epochs=epochs)

    n_samples, n_features = X_train.shape
    model.weights = np.zeros(n_features)
    model.bias = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, int(epochs) + 1):
        for X_batch, y_batch in _iterate_minibatches(
            X_train,
            y_train,
            batch_size=int(batch_size),
            shuffle=shuffle,
            random_state=random_state + epoch,
        ):
            linear_output = np.dot(X_batch, model.weights) + model.bias
            y_prob = model.sigmoid(linear_output)

            batch_n = X_batch.shape[0]
            dw = (1 / batch_n) * np.dot(X_batch.T, (y_prob - y_batch))
            db = (1 / batch_n) * np.sum(y_prob - y_batch)

            model.weights -= float(learning_rate) * dw
            model.bias -= float(learning_rate) * db

        train_prob = model.predict_probability(X_train)
        val_prob = model.predict_probability(X_val)

        train_loss = _binary_cross_entropy(y_train, train_prob)
        val_loss = _binary_cross_entropy(y_val, val_prob)

        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)

        train_acc = _accuracy(y_train, train_pred)
        val_acc = _accuracy(y_val, val_pred)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_accuracy"].append(round(train_acc * 100, 2))
        history["val_accuracy"].append(round(val_acc * 100, 2))

    y_val_pred = model.predict(X_val)

    split = {
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "train_ratio": float(train_ratio),
    }

    return model, history, split, y_val, y_val_pred
