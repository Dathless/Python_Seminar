# Workflow Report

# Diabetes Predictor ML System (Django + Manual Machine Learning Pipeline)

---

# 1. Tổng quan hệ thống

## 1.1 Giới thiệu

Tài liệu này mô tả toàn bộ luồng hoạt động (Workflow) của hệ thống dự đoán bệnh tiểu đường được xây dựng bằng Django kết hợp pipeline Machine Learning thủ công trong thư mục `core/`.

Mục tiêu của tài liệu bao gồm:

- Mô tả kiến trúc hoạt động của hệ thống
- Phân tích luồng xử lý dữ liệu
- Trình bày cơ chế request/response
- Mô tả pipeline huấn luyện và dự đoán
- Hỗ trợ xây dựng:
  - Sequence Diagram
  - Activity Diagram
  - Component Diagram
  - Deployment Diagram

Tài liệu này tập trung vào luồng vận hành của hệ thống ở mức kiến trúc và xử lý nghiệp vụ.

Chi tiết thuật toán học máy được trình bày riêng trong tài liệu `Algorithm.md`.

---

## 1.2 Phạm vi hệ thống

Hệ thống hiện bao gồm:

- Ứng dụng Web Django
- Dataset CSV cục bộ
- Pipeline Machine Learning thủ công
- Các chức năng:
  - Dataset Overview
  - Prediction Engine
  - Model Evaluation
  - Model Training

---

## 1.3 Các thành phần không thuộc phạm vi

Phiên bản hiện tại chưa triển khai:

- Authentication / Authorization
- REST API hoặc JSON API
- Lưu mô hình xuống Database
- Queue hoặc Background Training
- GPU Training thực tế
- Distributed Training

Toàn bộ quá trình huấn luyện hiện được thực thi trực tiếp theo từng request HTTP.

---

# 2. Kiến trúc hệ thống

## 2.1 Kiến trúc tổng thể

Hệ thống được tổ chức theo mô hình nhiều tầng gồm:

```text
Client Layer
    ↓
Django Web Layer
    ↓
Machine Learning Core Layer
    ↓
Dataset / Filesystem Layer
```

---

## 2.2 Các thành phần chính

### Client Layer

Bao gồm:

- Trình duyệt người dùng
- Giao diện HTML
- TailwindCSS CDN

Người dùng tương tác với hệ thống thông qua HTTP Request.

---

### Django Web Layer

Đây là tầng điều phối trung tâm của hệ thống.

Bao gồm:

| Thành phần          | Vai trò                  |
| ------------------- | ------------------------ |
| `config/urls.py`    | Root routing             |
| `diabetes/urls.py`  | Application routing      |
| `diabetes/views.py` | Xử lý request            |
| `diabetes/form.py`  | Validate dữ liệu đầu vào |
| `templates/`        | Render giao diện HTML    |

---

### Machine Learning Core Layer

Đây là tầng xử lý học máy của hệ thống.

Bao gồm:

| Module                        | Chức năng                  |
| ----------------------------- | -------------------------- |
| `core/data_processor.py`      | Tiền xử lý dữ liệu         |
| `core/predictor.py`           | Dự đoán và so sánh mô hình |
| `core/trainer.py`             | Huấn luyện mô hình         |
| `core/evaluator.py`           | Đánh giá mô hình           |
| `core/KNN.py`                 | KNN Manual                 |
| `core/decision_tree.py`       | Decision Tree Manual       |
| `core/logistic_regression.py` | Logistic Regression Manual |

---

### Dataset Layer

Nguồn dữ liệu chính:

```text
database/diabetes.csv
```

Dữ liệu được đọc trực tiếp từ filesystem trong quá trình xử lý request.

---

## 2.3 Actors của hệ thống

Các actor chính phục vụ việc xây dựng Sequence Diagram:

| Actor          | Vai trò                |
| -------------- | ---------------------- |
| User           | Người sử dụng hệ thống |
| Browser        | Gửi HTTP Request       |
| Django URLConf | Định tuyến URL         |
| Views          | Xử lý nghiệp vụ        |
| Forms          | Validate dữ liệu       |
| ML Core        | Xử lý Machine Learning |
| Filesystem     | Đọc dataset            |
| OS Metrics     | CPU/RAM/GPU metrics    |

---

# 3. Hệ thống Routing

## 3.1 Root Routing

File:

```text
config/urls.py
```

Các routing chính:

| URL       | Chức năng               |
| --------- | ----------------------- |
| `/admin/` | Django Admin            |
| `/`       | Include `diabetes.urls` |

---

## 3.2 Application Routing

File:

```text
diabetes/urls.py
```

---

## 3.3 Danh sách Endpoint

| URL                | Method   | View             | Chức năng                      |
| ------------------ | -------- | ---------------- | ------------------------------ |
| `/`                | GET      | `index`          | Dataset overview               |
| `/overview/`       | GET      | `index`          | Alias overview                 |
| `/predict/`        | GET/POST | `predict`        | Dự đoán bệnh                   |
| `/evaluation/`     | GET      | `evaluation`     | Đánh giá mô hình               |
| `/model_training/` | GET/POST | `model_training` | Huấn luyện Logistic Regression |

---

# 4. HTTP Request Lifecycle

## 4.1 Luồng xử lý tổng quát

Mỗi request trong hệ thống được xử lý theo quy trình:

```text
Browser
    ↓
Django URL Routing
    ↓
View Function
    ↓
Core Processing
    ↓
Build Context
    ↓
Render Template
    ↓
HTTP Response
```

---

## 4.2 Quy trình chi tiết

### Bước 1 — Client gửi request

Người dùng thao tác trên giao diện và gửi HTTP request thông qua trình duyệt.

---

### Bước 2 — URL Resolution

Django URLConf thực hiện ánh xạ URL đến view tương ứng.

---

### Bước 3 — Xử lý nghiệp vụ

View function sẽ:

- đọc dữ liệu
- validate input
- tiền xử lý dữ liệu
- huấn luyện mô hình
- dự đoán
- đánh giá

tùy theo endpoint được gọi.

---

### Bước 4 — Render Template

Dữ liệu được đóng gói thành `context dictionary` và render thành HTML response.

---

### Bước 5 — Trả response

HTML response được trả về trình duyệt người dùng.

---

# 5. Workflow Dataset Overview

## 5.1 Mục tiêu

Chức năng Dataset Overview được sử dụng để:

- hiển thị thông tin dataset
- preview dữ liệu
- thống kê mô tả
- kiểm tra cân bằng nhãn
- hiển thị dữ liệu chuẩn hóa

---

## 5.2 Endpoint

```text
GET /
GET /overview/
```

---

## 5.3 Input Parameters

| Parameter | Ý nghĩa                  |
| --------- | ------------------------ |
| `mode`    | original / normalized    |
| `all`     | hiển thị toàn bộ dữ liệu |

---

## 5.4 Quy trình xử lý

Workflow tổng quát:

```text
Load CSV
    ↓
Compute Statistics
    ↓
Compute Outcome Balance
    ↓
Normalize Dataset
    ↓
Select View Mode
    ↓
Render HTML
```

---

## 5.5 Các bước xử lý chi tiết

### Bước 1 — Đọc dataset

Hệ thống đọc:

```text
database/diabetes.csv
```

thông qua `pandas.read_csv()`.

---

### Bước 2 — Thống kê dữ liệu

Module:

```text
core.data_processor.field_statistic()
```

thực hiện:

- mean
- standard deviation
- missing percentage

---

### Bước 3 — Phân tích cân bằng dữ liệu

Module:

```text
core.data_processor.get_outcome_balance()
```

xác định tỷ lệ:

- Positive
- Negative

---

### Bước 4 — Chuẩn hóa dữ liệu

Dữ liệu được chuẩn hóa bằng phương pháp Z-score.

---

### Bước 5 — Chọn chế độ hiển thị

Người dùng có thể chọn:

- dữ liệu gốc
- dữ liệu chuẩn hóa

---

### Bước 6 — Render giao diện

Kết quả được render qua:

```text
overview.html
```

---

# 6. Workflow Prediction Engine

## 6.1 Mục tiêu

Prediction Engine cho phép:

- nhập thông tin bệnh nhân
- huấn luyện mô hình
- so sánh thuật toán
- thực hiện dự đoán bệnh tiểu đường

---

## 6.2 Endpoint

```text
GET /predict/
POST /predict/
```

---

## 6.3 Input Features

Các đặc trưng đầu vào:

| Feature                  |
| ------------------------ |
| Pregnancies              |
| Glucose                  |
| BloodPressure            |
| SkinThickness            |
| Insulin                  |
| BMI                      |
| DiabetesPedigreeFunction |
| Age                      |

---

## 6.4 Workflow tổng quát

```text
Validate Form
    ↓
Load Dataset
    ↓
Normalize Features
    ↓
Split Train/Test
    ↓
Train Models
    ↓
Compare Accuracy
    ↓
Select Best Model
    ↓
Predict Input
    ↓
Render Result
```

---

## 6.5 Quy trình xử lý chi tiết

### Bước 1 — Validate dữ liệu đầu vào

Module:

```text
diabetes.form.DataInput
```

thực hiện validate toàn bộ input người dùng.

---

### Bước 2 — Đọc dataset

Dataset được load từ filesystem.

---

### Bước 3 — Chuẩn hóa dữ liệu

Dữ liệu được chuẩn hóa bằng:

```text
core.data_processor.get_normalized_data()
```

---

### Bước 4 — Chia dữ liệu

Dataset được chia thành:

- train set
- test set

---

### Bước 5 — Khởi tạo mô hình

Hệ thống khởi tạo:

- KNN
- Decision Tree
- Logistic Regression

---

### Bước 6 — Huấn luyện mô hình

Mỗi mô hình được huấn luyện độc lập trên train set.

---

### Bước 7 — So sánh độ chính xác

Hệ thống đánh giá accuracy của từng mô hình trên test set.

---

### Bước 8 — Chọn mô hình tối ưu

Model có accuracy cao nhất sẽ được chọn làm kết quả chính.

---

### Bước 9 — Dự đoán dữ liệu mới

Input người dùng được chuẩn hóa và đưa vào mô hình để dự đoán.

---

### Bước 10 — Render kết quả

Kết quả dự đoán được hiển thị trên:

```text
predict.html
```

---

## 6.6 Edge Cases

### Invalid Input

Nếu dữ liệu không hợp lệ:

- form validation thất bại
- hệ thống không thực hiện huấn luyện

---

### Dataset Error

Nếu dataset không tồn tại hoặc lỗi:

- hệ thống raise exception
- context trả về chứa thông báo lỗi

---

### Performance Limitation

Mỗi request sẽ huấn luyện lại toàn bộ mô hình.

Điều này có thể gây:

- tăng thời gian phản hồi
- block request thread
- giảm hiệu năng khi dataset lớn

---

# 7. Workflow Evaluation

## 7.1 Mục tiêu

Trang Evaluation hiển thị:

- các chỉ số đánh giá
- confusion matrix metrics
- feature importance

---

## 7.2 Endpoint

```text
GET /evaluation/
```

---

## 7.3 Quy trình xử lý

Workflow:

```text
Load Dataset
    ↓
Generate Rule-Based Prediction
    ↓
Compute Metrics
    ↓
Compute Feature Importance
    ↓
Render Evaluation Page
```

---

## 7.4 Rule-Based Evaluation

Lưu ý:

Phiên bản hiện tại chưa đánh giá mô hình ML thực tế.

Rule demo hiện tại:

```text
Glucose > 125 → Positive
Ngược lại → Negative
```

---

## 7.5 Metrics được sử dụng

| Metric    |
| --------- |
| Accuracy  |
| Precision |
| Recall    |
| F1-score  |

---

## 7.6 Feature Importance

Feature importance được tính dựa trên hệ số tương quan giữa:

- feature
- Outcome

---

# 8. Workflow Model Training

## 8.1 Mục tiêu

Trang Model Training cho phép:

- huấn luyện Logistic Regression thủ công
- theo dõi training history
- quan sát validation metrics
- theo dõi system metrics

---

## 8.2 Endpoint

```text
GET /model_training/
POST /model_training/
```

---

## 8.3 Input Hyperparameters

| Parameter     | Default |
| ------------- | ------- |
| learning_rate | 0.01    |
| epochs        | 100     |
| batch_size    | 32      |
| train_ratio   | 80      |

---

## 8.4 Workflow tổng quát

```text
Collect System Metrics
    ↓
Load Dataset
    ↓
Normalize Data
    ↓
Split Train/Validation
    ↓
Mini-batch Training
    ↓
Validation Evaluation
    ↓
Build Training History
    ↓
Render Training Dashboard
```

---

## 8.5 System Metrics

Hệ thống hiển thị:

- CPU Usage
- RAM Usage
- GPU Availability

Thông tin được lấy từ:

- `psutil`
- `torch.cuda`

---

## 8.6 Training Pipeline

### Bước 1 — Chuẩn hóa dữ liệu

Dataset được chuẩn hóa bằng Z-score.

---

### Bước 2 — Chia dữ liệu

Dataset được chia:

- training set
- validation set

---

### Bước 3 — Mini-batch Gradient Descent

Logistic Regression được huấn luyện bằng:

```text
Mini-batch Gradient Descent
```

---

### Bước 4 — Validation

Mô hình được đánh giá trên validation set.

---

### Bước 5 — Lưu training history

Hệ thống lưu:

- train loss
- validation loss
- train accuracy
- validation accuracy

theo từng epoch.

---

### Bước 6 — Render dashboard

Kết quả được hiển thị trên:

```text
model_training.html
```

---

## 8.7 Edge Cases

### Large Epochs

Epoch quá lớn có thể:

- làm chậm hệ thống
- block request thread

---

### Runtime Error

Nếu xảy ra exception:

- hệ thống trả HTTP 500
- response chứa thông báo lỗi

---

# 9. Data Flow

## 9.1 Dataset Schema

Dataset bao gồm:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

---

## 9.2 Data Normalization

Toàn bộ feature được chuẩn hóa bằng:

```text
Z-score Normalization
```

ngoại trừ:

```text
Outcome
```

---

## 9.3 Train/Test Split

Hệ thống sử dụng:

- Train/Test Split
- Train/Validation Split

tùy theo workflow.

---

# 10. Component Responsibilities

## 10.1 Routing Layer

| File               | Vai trò      |
| ------------------ | ------------ |
| `config/urls.py`   | Root routing |
| `diabetes/urls.py` | App routing  |

---

## 10.2 Controller Layer

| File                | Vai trò                 |
| ------------------- | ----------------------- |
| `diabetes/views.py` | HTTP request processing |

---

## 10.3 Form Layer

| File               | Vai trò          |
| ------------------ | ---------------- |
| `diabetes/form.py` | Input validation |

---

## 10.4 Template Layer

| Template              |
| --------------------- |
| `overview.html`       |
| `predict.html`        |
| `evaluation.html`     |
| `model_training.html` |

---

## 10.5 Machine Learning Layer

| Module              | Vai trò              |
| ------------------- | -------------------- |
| `data_processor.py` | Tiền xử lý           |
| `predictor.py`      | Prediction pipeline  |
| `trainer.py`        | Training pipeline    |
| `evaluator.py`      | Metrics & evaluation |

---

# 11. Kết luận

Hệ thống được xây dựng theo kiến trúc phân tầng giữa:

- giao diện người dùng
- tầng xử lý web Django
- tầng Machine Learning
- tầng dữ liệu

Pipeline Machine Learning được triển khai thủ công nhằm phục vụ:

- nghiên cứu thuật toán
- học thuật
- phân tích nguyên lý hoạt động của mô hình

Workflow hiện tại hỗ trợ đầy đủ:

- tiền xử lý dữ liệu
- huấn luyện mô hình
- đánh giá mô hình
- dự đoán dữ liệu mới
- trực quan hóa kết quả

Kiến trúc hệ thống được tổ chức theo hướng mô-đun hóa giúp:

- dễ mở rộng
- dễ bảo trì
- thuận tiện cho việc nâng cấp thuật toán trong tương lai.
