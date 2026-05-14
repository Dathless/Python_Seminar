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
from core.trainer import train_model_process
BASE_DIR = Path(__file__).resolve().parent.parent
from .form import DataInput
import numpy as np
import pandas as pd
from core.decision_tree import DecisionTreeManual
from core.KNN import KNN
from core.predictor import load_all_model_and_compare
import psutil
import torch
import re



# Create your views here.

def index(request):
    file_path = BASE_DIR / "database" / "diabetes.csv"
    data = load_data(file_path)
    statistics = field_statistic(data)
    outcome_balance = get_outcome_balance(data)
    normalized_data, normalization_info = get_normalized_data(data)

    view_mode = request.GET.get("mode", "normalized").lower()
    view_all = request.GET.get("all", "0") == "1"

    if view_mode == "original":
        data_for_view = data
    else:
        view_mode = "normalized"
        data_for_view = normalized_data

    if view_all:
        data_for_preview = data_for_view
    else:
        data_for_preview = data_for_view.head(10)

    preview_data = data_for_preview.to_dict(orient="records")

    context = {
        "total_records": len(data),
        "preview_data": preview_data,
        "showing_count": len(data_for_preview),
        "view_mode": view_mode,
        "view_all": view_all,
        "statistics": statistics,
        "outcome_balance": outcome_balance
    }

    return render(
        request,
        'diabetes/overview.html',
        context
    )

def predict(request):
    if request.method == 'POST':
        form = DataInput(request.POST)
        
        if form.is_valid():
            # In debug dữ liệu hợp lệ
            print("--- Dữ liệu nhận được từ Form ---")
            print(form.cleaned_data) 
            
            try:
                # 1. Load dữ liệu (Đảm bảo đúng đường dẫn file)
                data = pd.read_csv('database/diabetes.csv') 
                X = data.drop('Outcome', axis=1).values
                y = data['Outcome'].values
                
                # 2. Huấn luyện model
                model_dt = DecisionTreeManual(max_depth=5, min_samples_split=2)
                model_knn = KNN(k=5)
                model_dt.fit(X, y)
                model_knn.fit(X, y)

                # 3. Lấy dữ liệu sạch từ form
                clean_form = form.cleaned_data
                input_data = np.array([[
                    float(clean_form['pregnancies']), float(clean_form['glucose']), 
                    float(clean_form['blood_pressure']), float(clean_form['skin_thickness']), 
                    float(clean_form['insulin']), float(clean_form['bmi']), 
                    float(clean_form['diabetes_pedigree_function']), float(clean_form['age'])
                ]])
                
                # 4. Dự đoán
                prediction_result = model_dt.predict(input_data)
                knn_predict = model_knn.predict(input_data)
                prediction = int(prediction_result[0])
                result = load_all_model_and_compare( X, y,input_data)
                print(f"--- Dự đoán kết quả: {prediction} --- | Decision Tree")
                print(f"--- Dự đoán kết quả: {knn_predict[0]} --- | KNN")
                print("=" * 40)
                print("MODEL PREDICTION REPORT")
                print("=" * 40)

                print("\nPredictions:")
                for model, pred in result["predictions"].items():
                    print(f"  {model:<20}: {pred}")

                print("\nAccuracies:")
                for model, acc in result["accuracies"].items():
                    print(f"  {model:<20}: {acc}%")

                print("\nDataset Split:")
                print(f"  Train Ratio         : {result['train_ratio']}")
                print(f"  Test Ratio          : {result['test_ratio']}")

                print("=" * 40)

            except Exception as e:
                print(f"--- LỖI HỆ THỐNG: {e} ---")
                # Nếu có lỗi trong quá trình tính toán, in ra terminal thay vì trả về 500
        else:
            print("--- Lỗi Form ---")
            print(form.errors)

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
    try:
        # --- 1. LẤY THÔNG SỐ HỆ THỐNG ---
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # RAM: Tránh lỗi chia cho 0 hoặc lỗi truy cập hệ thống
        vm = psutil.virtual_memory()
        current_mem = round(vm.used / (1024**3), 1)
        max_mem = round(vm.total / (1024**3), 1)
        memory_usage_str = f"{current_mem}/{max_mem} GB"
        
        # GPU
        if torch.cuda.is_available():
            full_gpu_name = torch.cuda.get_device_name(0)
            match = re.search(r"RTX\s?\d{4}[^\s]*", full_gpu_name)
            short_name = match.group(0) if match else full_gpu_name
            
            if request.method == "POST":
                gpu_status_code = "ACTIVE"
                gpu_acceleration = f"ACTIVE ({short_name})"
            else:
                gpu_status_code = "INACTIVE"
                gpu_acceleration = f"INACTIVE ({short_name})"
        else:
            gpu_acceleration = "--"

        # --- 2. LOAD DỮ LIỆU ---
        # Chú ý: Đảm bảo BASE_DIR đã được định nghĩa ở đầu file views.py
        file_path = BASE_DIR / "database" / "diabetes.csv"
        data = load_data(file_path)
        normalized_data, _ = get_normalized_data(data)

        X = normalized_data.drop("Outcome", axis=1).values
        y = normalized_data["Outcome"].values.astype(int)

        # Khởi tạo context mặc định
        context = {
            "trained": False,
            "total_records": len(data),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage_str,
            "gpu_acceleration": gpu_acceleration,
            "gpu_status_code": gpu_status_code,
            "default_lr": 0.01,
            "default_epochs": 100,
            "default_batch_size": 32,
            "default_train_ratio": 80,
            "final_val_acc": "--",
            "final_val_loss": "--",
            "metrics": {"f1_score": "--"},
            "history": {"epoch": [], "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []},
        }

        if request.method == "POST":
            learning_rate = float(request.POST.get("learning_rate", 0.01))
            epochs = int(request.POST.get("epochs", 100))
            batch_size = int(request.POST.get("batch_size", 32))
            train_ratio_percent = int(request.POST.get("train_ratio", 80))
            train_ratio = max(50, min(90, train_ratio_percent)) / 100.0

            # Giả sử hàm train_model_process trả về dữ liệu đúng cấu trúc
            _model, history, split, y_val, y_val_pred = train_model_process(
                X, y, learning_rate=learning_rate, epochs=epochs, 
                batch_size=batch_size, train_ratio=train_ratio
            )

            metrics = evaluate_model(y_val, y_val_pred)

            # Kiểm tra history có dữ liệu không trước khi lấy phần tử cuối [-1] để tránh IndexError
            has_history = history and history.get("train_loss")

            context.update({
                "trained": True,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_ratio": train_ratio_percent,
                "test_ratio": 100 - train_ratio_percent,
                "history": history,
                "split": split,
                "metrics": metrics,
                "final_train_loss": round(history["train_loss"][-1], 4) if has_history else "--",
                "final_val_loss": round(history["val_loss"][-1], 4) if has_history else "--",
                "final_train_acc": round(history["train_accuracy"][-1], 2) if has_history else "--",
                "final_val_acc": round(history["val_accuracy"][-1], 2) if has_history else "--",
            })
        else:
            # Case GET request
            context.update({
                "learning_rate": context["default_lr"],
                "epochs": context["default_epochs"],
                "batch_size": context["default_batch_size"],
                "train_ratio": context["default_train_ratio"],
                "test_ratio": 100 - context["default_train_ratio"],
            })

        return render(request, "diabetes/model_training.html", context)

    except Exception as e:
        # In lỗi ra console để debug nếu vẫn bị 500
        print(f"Error in model_training: {e}")
        from django.http import HttpResponse
        return HttpResponse(f"Đã xảy ra lỗi hệ thống: {e}", status=500)
