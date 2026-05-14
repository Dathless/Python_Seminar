import numpy as np
import pandas as pd

from core.KNN import KNN
from core.decision_tree import DecisionTreeManual
from core.data_processor import get_normalized_data
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

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _ensure_2d_array(input_data):
    array = np.asarray(input_data, dtype=float)

    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array

    raise ValueError(
        "Input data must be a 1D or 2D array-like object for prediction."
    )


def _normalize_train_ratio(train_ratio):
    if train_ratio is None:
        return 0.8

    if isinstance(train_ratio, (int, np.integer)) and train_ratio > 1:
        train_ratio = float(train_ratio) / 100.0

    if not isinstance(train_ratio, float):
        train_ratio = float(train_ratio)

    if train_ratio <= 0 or train_ratio >= 1:
        return 0.8

    return train_ratio


def _prepare_normalized_features(X, y):
    if isinstance(X, pd.DataFrame):
        data_frame = X.copy()
        if "Outcome" not in data_frame.columns:
            data_frame["Outcome"] = y
    else:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if X_arr.ndim != 2:
            raise ValueError("Feature matrix X must be 2D.")

        column_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        data_frame = pd.DataFrame(X_arr, columns=column_names)
        data_frame["Outcome"] = y_arr

    normalized_df, normalization_info = get_normalized_data(data_frame)
    X_norm = normalized_df.drop("Outcome", axis=1).values
    y_norm = normalized_df["Outcome"].values.astype(int)
    return X_norm, y_norm, normalization_info


def _normalize_input_data(input_data, normalization_info):
    array = _ensure_2d_array(input_data)
    if array.shape[1] != len(normalization_info):
        raise ValueError(
            "Input data feature count does not match normalized model feature count."
        )

    normalized_array = np.copy(array)
    feature_names = list(normalization_info.keys())

    for idx, feat_name in enumerate(feature_names):
        mean_value = normalization_info[feat_name]["mean"]
        std_value = normalization_info[feat_name]["std"]
        if std_value != 0:
            normalized_array[:, idx] = (
                normalized_array[:, idx] - mean_value
            ) / std_value

    return normalized_array


def load_all_models(
    knn_k=5,
    dt_max_depth=5,
    dt_min_samples_split=2,
    lr_learning_rate=0.01,
    lr_epochs=100,
):
    knn_model = KNN(k=knn_k)
    decision_tree_model = DecisionTreeManual(
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
    )
    logistic_regression_model = LogisticRegressionManual(
        learning_rate=lr_learning_rate,
        epochs=lr_epochs,
    )

    return knn_model, decision_tree_model, logistic_regression_model


def load_all_model_and_compare(
    X,
    y,
    input_data,
    *,
    train_ratio=0.8,
    random_state=42,
    knn_k=5,
    dt_max_depth=5,
    dt_min_samples_split=2,
    lr_learning_rate=0.01,
    lr_epochs=100,
    shuffle=True,
):
    """Train and compare KNN, Decision Tree, and Logistic Regression.

    Returns a dict with model predictions for the provided input and test-set accuracies.
    If train_ratio is invalid or missing, it falls back to the default 80/20 split.
    """

    train_ratio = _normalize_train_ratio(train_ratio)
    test_ratio = round(1.0 - train_ratio, 2)

    X_norm, y_norm, normalization_info = _prepare_normalized_features(X, y)

    X_train, X_test, y_train, y_test = _train_test_split(
        X_norm,
        y_norm,
        train_ratio=train_ratio,
        shuffle=shuffle,
        random_state=random_state,
    )

    knn_model, dt_model, lr_model = load_all_models(
        knn_k=knn_k,
        dt_max_depth=dt_max_depth,
        dt_min_samples_split=dt_min_samples_split,
        lr_learning_rate=lr_learning_rate,
        lr_epochs=lr_epochs,
    )

    knn_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    normalized_input = _normalize_input_data(input_data, normalization_info)
    prediction_knn = int(knn_model.predict(normalized_input)[0])
    prediction_dt = int(dt_model.predict(normalized_input)[0])
    prediction_lr = int(lr_model.predict(normalized_input)[0])

    return {
        "predictions": {
            "knn": prediction_knn,
            "decision_tree": prediction_dt,
            "logistic_regression": prediction_lr,
        },
        "accuracies": {
            "knn": round(_accuracy(y_test, y_pred_knn) * 100, 2),
            "decision_tree": round(_accuracy(y_test, y_pred_dt) * 100, 2),
            "logistic_regression": round(_accuracy(y_test, y_pred_lr) * 100, 2),
        },
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "normalization_info": normalization_info,
    }
