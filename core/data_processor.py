import pandas as pd
import numpy as np


CLINICAL_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# đọc file csv, tính thống kê, tỉ lệ positive/negative và chuẩn hóa dữ liệu
def load_data(file_path):
    """
    Load diabetes dataset
    """
    data = pd.read_csv(file_path)
    return data

# tính mean (Giá trị trung bình), std (Độ lệch chuẩn) và missing % (% Dữ liệu trống) của các cột dữ liệu (trừ cột Outcome)
def field_statistic(data):
    """
    Tính mean, std và missing %
    """

    statistics = []

    for column in data.columns:

        if column == "Outcome":
            continue

        mean_value = round(data[column].mean(), 2)
        std_value = round(data[column].std(), 2)

        # Missing value = 0 ở các cột lâm sàng
        if column in CLINICAL_COLUMNS:
            missing_count = (data[column] == 0).sum()
        else:
            missing_count = 0

        missing_percent = round(
            (missing_count / len(data)) * 100,
            2
        )

        statistics.append({
            "field": column,
            "mean": mean_value,
            "std": std_value,
            "missing_percent": missing_percent
        })

    return statistics

# đếm số lượng bản ghi Positive và Negative, tính tỉ lệ phần trăm của chúng trong tổng số bản ghi
def get_outcome_balance(data):
    """
    Tính tỉ lệ Positive / Negative
    """

    total = len(data)

    positive = (data["Outcome"] == 1).sum()
    negative = (data["Outcome"] == 0).sum()

    positive_percent = round((positive / total) * 100, 2)
    negative_percent = round((negative / total) * 100, 2)

    return {
        "positive": positive,
        "negative": negative,
        "positive_percent": positive_percent,
        "negative_percent": negative_percent
    }

# chuẩn hóa dữ liệu bằng phương pháp Z-score, tính mean và std của từng cột dữ liệu 
# (trừ cột Outcome) để chuẩn hóa, trả về dữ liệu đã chuẩn hóa và 
# thông tin về mean và std của từng cột
def get_normalized_data(data):
    """
    Chuẩn hóa Z-score
    """

    normalized_data = data.copy()

    feature_columns = [
        col for col in data.columns
        if col != "Outcome"
    ]

    normalization_info = {}

    for column in feature_columns:

        mean_value = data[column].mean()
        std_value = data[column].std()

        normalization_info[column] = {
            "mean": mean_value,
            "std": std_value
        }

        if std_value != 0:
            normalized_data[column] = (
                (data[column] - mean_value) / std_value
            )

    return normalized_data.round(3), normalization_info