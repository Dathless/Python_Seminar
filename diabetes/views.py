from django.shortcuts import render
from pathlib import Path
from core.evaluator import (
    evaluate_model,
    get_feature_importance
)

from core.data_processor import (
    load_data,
    field_statistic,
    get_outcome_balance,
    get_normalized_data
)

BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):
    file_path = BASE_DIR / "database" / "diabetes.csv"
    data = load_data(file_path)
    statistics = field_statistic(data)
    outcome_balance = get_outcome_balance(data)
    normalized_data, normalization_info = get_normalized_data(data)
    preview_data = normalized_data.head(10).to_dict(orient="records")

    context = {
        "total_records": len(data),
        "preview_data": preview_data,
        "statistics": statistics,
        "outcome_balance": outcome_balance
    }

    return render(
        request,
        'diabetes/overview.html',
        context
    )

def predict(request):
    return render(request, 'diabetes/predict.html')

def evaluation(request):
    file_path = BASE_DIR / "database" / "diabetes.csv"

    data = load_data(file_path)

    # Nhãn thật
    y_true = data["Outcome"]

    # Rule-based prediction demo
    # Nếu Glucose > 125 => Positive
    y_pred = []

    for glucose in data["Glucose"]:

        if glucose > 125:
            y_pred.append(1)
        else:
            y_pred.append(0)

    metrics = evaluate_model(y_true, y_pred)
    feature_importance = get_feature_importance(data)

    context = {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "total_records": len(data)
    }

    return render(
        request,
        'diabetes/evaluation.html',
        context
    )

def model_training(request):
    return render(request, 'diabetes/model_training.html')