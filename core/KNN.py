# main.py hoặc train.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.KNN import KNN

# Đọc diabetes.csv

df = pd.read_csv("dataset/diabetes.csv")

# Features
X = df.drop("Outcome", axis=1).values

# Label
y = df["Outcome"].values

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
model = KNN(k=5)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = model.accuracy(y_test, predictions)

print("Accuracy:", accuracy)