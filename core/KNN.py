# core/KNN.py

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    # Lưu dữ liệu train
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Tính khoảng cách Euclidean
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Predict 1 sample
    def predict_one(self, x):
        distances = []

        # Tính khoảng cách từ sample tới toàn bộ train data
        for x_train in self.X_train:
            distance = self.euclidean_distance(x, x_train)
            distances.append(distance)

        # Lấy index của k điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]

        # Lấy label tương ứng
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Voting
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]

    # Predict nhiều sample
    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.predict_one(x)
            predictions.append(prediction)

        return np.array(predictions)

    # Accuracy
    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)