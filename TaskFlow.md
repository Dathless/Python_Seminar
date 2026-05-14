# Công việc

# 📋 PHÂN TÍCH CẤU TRÚC HỆ THỐNG ML MANUAL - DIABETES PREDICTION

## 📂 1. Cấu trúc thư mục (Project Architecture)

Hệ thống được tổ chức theo mô hình tách biệt logic (Decoupling), giúp dễ dàng bảo trì và mở rộng thuật toán mà không ảnh hưởng đến cấu hình Web.

```text
diabetes_project/
│
├── config/                 # Root project (Tạo bởi startproject)
│   ├── settings.py         # Cấu hình hệ thống, installed_apps
│   ├── urls.py             # Điều phối URL tổng của dự án
│   └── wsgi.py / asgi.py
│
├── core/                   # Lõi xử lý logic Machine Learning (Manual)
│   ├── __init__.py         # Đánh dấu thư mục là Python Package
│   ├── data_processor.py   # Load dữ liệu, tính thống kê và Z-score (rồi)
│   ├── knn.py              # Thuật toán K-Nearest Neighbors manual (rồi)
│   ├── decision_tree.py    # Thuật toán Decision Tree manual
│   ├── logistics_reg.py    # Thuật toán Logistic Regression manual (rồi)
│   ├── trainer.py          # Logic huấn luyện (Epoch, Batch, Loss history)
│   └── evaluator.py        # Tính toán ma trận nhầm lẫn và Metrics (rồi)
│
├── diabetes/               # [startapp] Django App (Giao diện & API)
│   ├── migrations/         
│   ├── templates/          # Chứa các file HTML giao diện
│   ├── __init__.py
│   ├── forms.py            # Form nhận dữ liệu lâm sàng từ người dùng
│   ├── views.py            # Gọi logic từ core/ để trả kết quả lên UI
│   ├── urls.py             # URL nội bộ của app diabetes
│   └── admin.py
│
├── dataset/                # Nơi lưu trữ dữ liệu nguồn
│   └── diabetes.csv        # Pima Indians Diabetes Dataset
│
├── manage.py               # File thực thi chính của Django
└── requirements.txt        # Danh sách thư viện (Pandas, Numpy, Django)
```

## 🛠️ 2. Chi tiết các File và Hàm chức năng (Technical Specifications)

Phần này định nghĩa logic nghiệp vụ cho từng module trong hệ thống. Các thuật toán Machine Learning được yêu cầu triển khai thủ công (Manual) không sử dụng thư viện tích hợp sẵn như Scikit-learn để đảm bảo tính tùy biến và mục tiêu học thuật.

### 📊 A. File: `data_processor.py`
**Mục tiêu:** Tiền xử lý dữ liệu thô và cung cấp thông số thống kê cho Dashboard.

* `load_data(file_path)`: 
    * **Chức năng:** Sử dụng thư viện `pandas` hoặc `csv` để đọc tập dữ liệu Pima Indians Diabetes.
    * **Đầu ra:** DataFrame hoặc List of Dictionaries.
* `field_statistic(data)`: 
    * **Chức năng:** Duyệt qua các cột (Glucose, BloodPressure, BMI...), tính toán giá trị trung bình (`mean`), độ lệch chuẩn (`std dev`).
    * **Xử lý Missing Value:** Đếm các giá trị bằng `0` ở các cột lâm sàng (vốn là dữ liệu thiếu) để tính tỷ lệ `% Missing`.
* `get_outcome_balance(data)`: 
    * **Chức năng:** Đếm số lượng mẫu `Positive` (nhãn 1) và `Negative` (nhãn 0).
    * **Đầu ra:** Trả về tỷ lệ phần trăm (ví dụ: 34.9% Positive) để hiển thị biểu đồ "Outcome Balance".
* `get_normalized_data(data)`: 
    * **Chức năng:** Thực hiện chuẩn hóa Z-score theo công thức: $z = \frac{x - \mu}{\sigma}$.
    * **Ghi chú:** Phải lưu trữ lại giá trị $\mu$ (mean) và $\sigma$ (std) của tập huấn luyện để chuẩn hóa dữ liệu đầu vào khi dự đoán thực tế.

---

### 🧠 B. File: `models_manual.py`
**Mục tiêu:** Xây dựng lõi thuật toán học máy từ các công thức toán học cơ bản.

* **Class `LogisticRegressionManual`**: 
    * Sử dụng hàm kích hoạt **Sigmoid**: $f(z) = \frac{1}{1 + e^{-z}}$.
    * Cập nhật trọng số thông qua thuật toán **Gradient Descent**.
* **Class `KNNManual`**: 
    * Tính toán khoảng cách **Euclidean** giữa điểm mới và các điểm trong tập huấn luyện.
    * Tìm $K$ láng giềng gần nhất để thực hiện biểu quyết (voting) kết quả.
* **Class `DecisionTreeManual`**: 
    * Sử dụng chỉ số **Gini Impurity** hoặc **Entropy** để đo lường độ vẩn đục của dữ liệu.
    * Thực hiện đệ quy phân nhánh cây dựa trên **Information Gain** cho đến khi đạt `max_depth` hoặc nút lá thuần khiết.
* `load_all_models_and_compare()`: 
    * **Chức năng:** Khởi tạo và chạy thử cả 3 model trên cùng một tập dữ liệu.
    * **Đầu ra:** Trả về đối tượng chứa tên model, kết quả dự đoán và độ chính xác (accuracy) tương ứng.

---

### 🚀 C. File: `trainer.py`
**Mục tiêu:** Quản lý vòng lặp huấn luyện và theo dõi hiệu suất mô hình theo thời gian thực.

* `train_model_process(params)`: 
    * **Input:** Nhận bộ tham số từ HTML (Learning Rate, Epochs, Batch Size).
    * **Cơ chế:** Chia dữ liệu thành các **Mini-batches** để huấn luyện nhằm tăng tốc độ tính toán.
* `capture_training_history()`: 
    * **Chức năng:** Sau mỗi Epoch, ghi lại giá trị `Loss` (mất mát) và `Accuracy` (độ chính xác) của cả tập Train và Validation.
    * **Đầu ra:** Một mảng JSON (History) dùng để vẽ biểu đồ đường "Training Loss vs Validation Loss".

#### ✅ Trạng thái triển khai (Project hiện tại)

- `core/trainer.py`: đã triển khai `train_model_process(...)` cho Logistic Regression manual (mini-batch, split train/val, trả về `history` để render UI).
- `diabetes/views.py`: route `model_training` nhận POST từ `model_training.html` để chạy train và trả về `metrics` + `history`.

---

### 📈 D. File: `evaluator.py`
**Mục tiêu:** Tính toán các chỉ số kiểm định chất lượng sau khi mô hình đã hoàn tất huấn luyện.

* `get_confusion_matrix(y_true, y_pred)`: 
    * **Chức năng:** Tính toán ma trận nhầm lẫn 2x2 gồm: **True Positive (TP)**, **False Positive (FP)**, **False Negative (FN)**, **True Negative (TN)**.
* `calculate_metrics(matrix)`: 
    * **Chức năng:** Tính toán bộ chỉ số:
        * $Precision = \frac{TP}{TP + FP}$
        * $Recall = \frac{TP}{TP + FN}$
        * $F1\text{-}Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
* `get_feature_importance(model)`: 
    * **Chức năng:** Xác định mức độ đóng góp của từng đặc trưng (Glucose, BMI, Age...) vào kết quả cuối cùng. Trả về mảng dữ liệu để hiển thị biểu đồ thanh "Feature Importance".
