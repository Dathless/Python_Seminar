from django.shortcuts import render
from .form import DataInput
import numpy as np
import pandas as pd
from core.decision_tree import DecisionTreeManual

# Create your views here.

def index(request):
    return render(request, 'diabetes/overview.html')

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
                model_dt.fit(X, y)
                
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
                prediction = int(prediction_result[0])
                print(f"--- Dự đoán kết quả: {prediction} ---")
                
            except Exception as e:
                print(f"--- LỖI HỆ THỐNG: {e} ---")
                # Nếu có lỗi trong quá trình tính toán, in ra terminal thay vì trả về 500
        else:
            print("--- Lỗi Form ---")
            print(form.errors)

    return render(request, 'diabetes/predict.html')

def evaluation(request):
    return render(request, 'diabetes/evaluation.html')

def model_training(request):
    return render(request, 'diabetes/model_training.html')