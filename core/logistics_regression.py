import numpy as np

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        learning_rate : tốc độ cập nhật weight
        epochs        : số lần học toàn bộ dataset
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        #weights và bias sẽ được khởi tạo sau khi biết số lượng đặc trưng của dữ liệu
        self.weights = None
        self.bias = 0

        # Lưu lịch sử training
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, z):
        """
        Hàm sigmoid:
            f(z) = 1 / (1 + e^-z)

        Chuyển output về xác suất nằm trong khoảng 0 -> 1
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross Entropy Loss
        """

        # Tránh log(0)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return loss
    
    def fit(self, X, y):
        """
        Huấn luyện model bằng Gradient Descent
        """

        # số sample và số feature
        n_samples, n_features = X.shape

        # Khởi tạo weight = 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            # Linear model
            # z = wx + b
            linear_output = np.dot(X, self.weights) + self.bias

            # Áp dụng sigmoid để có xác suất
            y_pred = self.sigmoid(linear_output)

            # Gradient Calculation
            # dw = (1/n) * X^T(y_pred - y): đạo hàm theo weight của loss function
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))

            # db = (1/n) * sum(y_pred - y): đạo hàm theo bias của loss function
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Cập nhật weights và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Tính loss và accuracy để theo dõi quá trình học
            loss = self.compute_loss(y, y_pred)
            predictions = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

    def predict_probability(self, X):
        """
        Trả về xác suất bị bệnh
        """

        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)
    
    def predict(self, X):
        """
        Trả về nhãn dự đoán (0 hoặc 1):
            >= 0.5 -> 1: bị bệnh
            < 0.5  -> 0: không bị bệnh
        """

        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)
    
    def get_params(self):
        """
        Lấy weight và bias hiện tại
        """

        return {
            "weights": self.weights,
            "bias": self.bias
        }