# Algorithm Report

## 1. Tổng quan hệ thống thuật toán

Tài liệu này mô tả chi tiết các thuật toán, cơ chế xử lý dữ liệu và quy trình huấn luyện mô hình được triển khai trong thư mục `core/` của hệ thống dự đoán bệnh tiểu đường.

Các mô-đun chính bao gồm:

- `core/data_processor.py`
- `core/KNN.py`
- `core/decision_tree.py`
- `core/logistic_regression.py`
- `core/trainer.py`
- `core/evaluator.py`
- `core/predictor.py`

Hệ thống được xây dựng theo pipeline xử lý dữ liệu và học máy gồm các giai đoạn:

1. Tiền xử lý và chuẩn hóa dữ liệu
2. Huấn luyện mô hình học máy
3. Đánh giá hiệu năng mô hình
4. So sánh thuật toán
5. Dự đoán dữ liệu đầu vào mới

Các thuật toán học máy được triển khai thủ công nhằm phục vụ mục tiêu nghiên cứu, học thuật và phân tích nguyên lý hoạt động của từng mô hình.

---

# 2. Kiến trúc xử lý dữ liệu

## 2.1 Module `core/data_processor.py`

### 2.1.1 Mục tiêu

Mô-đun `data_processor.py` chịu trách nhiệm xử lý dữ liệu đầu vào, thống kê dữ liệu và chuẩn hóa đặc trưng trước khi đưa vào quá trình huấn luyện mô hình.

Đây là tầng tiền xử lý dữ liệu (Data Preprocessing Layer) trong hệ thống.

---

### 2.1.2 Các chức năng chính

#### `load_data(file_path)`

Hàm thực hiện đọc dữ liệu từ file CSV bằng thư viện `pandas`.

Dữ liệu sau khi đọc được lưu dưới dạng `DataFrame` nhằm hỗ trợ thao tác thống kê và xử lý đặc trưng.

---

#### `field_statistic(data)`

Hàm thực hiện thống kê mô tả cho từng thuộc tính trong tập dữ liệu.

Các thông số được tính toán bao gồm:

- Giá trị trung bình (`mean`)
- Độ lệch chuẩn (`std`)
- Tỷ lệ dữ liệu thiếu (`missing_percent`)

Đối với các thuộc tính lâm sàng:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

giá trị bằng `0` được xem là dữ liệu thiếu do không phù hợp trong ngữ cảnh y khoa.

Cột `Outcome` được loại khỏi quá trình thống kê vì đây là nhãn phân loại của mô hình.

---

#### `get_outcome_balance(data)`

Hàm xác định mức độ cân bằng dữ liệu giữa hai lớp phân loại:

- `Outcome = 1`: dương tính
- `Outcome = 0`: âm tính

Kết quả bao gồm:

- số lượng mẫu của từng lớp
- tỷ lệ phần trăm của từng lớp trong toàn bộ tập dữ liệu

Thông tin này giúp đánh giá nguy cơ mất cân bằng dữ liệu (imbalanced dataset).

---

#### `get_normalized_data(data)`

Hàm thực hiện chuẩn hóa dữ liệu bằng phương pháp Z-score:

```math
z = \frac{x - \mu}{\sigma}
```

Trong đó:

- `x`: giá trị gốc
- `μ`: giá trị trung bình
- `σ`: độ lệch chuẩn

Việc chuẩn hóa giúp:

- đưa các đặc trưng về cùng thang đo
- giảm ảnh hưởng của đơn vị đo
- hỗ trợ các thuật toán phụ thuộc khoảng cách hoặc gradient

Hàm đồng thời lưu lại thông tin:

- `mean`
- `std`

để phục vụ chuẩn hóa dữ liệu mới trong giai đoạn suy luận.

---

### 2.1.3 Vai trò trong hệ thống

`data_processor.py` đóng vai trò là tầng tiền xử lý trung tâm của toàn bộ hệ thống.

Mọi dữ liệu trước khi huấn luyện hoặc dự đoán đều phải trải qua bước chuẩn hóa nhằm đảm bảo tính nhất quán và ổn định của mô hình.

---

# 3. Thuật toán K-Nearest Neighbors

## 3.1 Module `core/KNN.py`

### 3.1.1 Mục tiêu

Mô-đun triển khai thuật toán K-Nearest Neighbors (KNN) phục vụ bài toán phân loại nhị phân.

Thuật toán hoạt động dựa trên giả định:

> Các mẫu dữ liệu có đặc trưng tương đồng thường thuộc cùng một lớp.

---

### 3.1.2 Cấu trúc mô hình

#### `class KNN`

Lớp `KNN` quản lý toàn bộ quá trình huấn luyện và dự đoán của mô hình.

Các tham số chính:

- `k`: số lượng láng giềng gần nhất được sử dụng trong quá trình bỏ phiếu

---

### 3.1.3 Quy trình huấn luyện

#### `fit(X, y)`

Do KNN là thuật toán học máy phi tham số (Non-parametric Learning Algorithm), quá trình huấn luyện không thực hiện tối ưu trọng số.

Hàm `fit()` chỉ lưu trữ:

- tập đặc trưng huấn luyện `X`
- tập nhãn `y`

để sử dụng trong giai đoạn suy luận.

---

### 3.1.4 Tính khoảng cách Euclid

#### `euclidean_distance(x1, x2)`

Khoảng cách giữa hai vector đặc trưng được tính theo công thức:

```math
d = \sqrt{\sum_{i=1}^{n}(x1_i - x2_i)^2}
```

Khoảng cách Euclid phản ánh mức độ tương đồng giữa hai mẫu dữ liệu trong không gian nhiều chiều.

Khoảng cách càng nhỏ cho thấy mức độ tương đồng càng cao.

---

### 3.1.5 Quy trình dự đoán

#### `predict_one(x)`

Quá trình dự đoán cho một mẫu dữ liệu diễn ra theo các bước:

1. Tính khoảng cách từ mẫu đầu vào đến toàn bộ dữ liệu huấn luyện
2. Sắp xếp khoảng cách theo thứ tự tăng dần
3. Chọn `k` mẫu gần nhất
4. Thực hiện cơ chế majority voting
5. Trả về nhãn xuất hiện nhiều nhất

---

#### `predict(X)`

Hàm thực hiện suy luận trên toàn bộ tập dữ liệu đầu vào bằng cách lặp qua từng mẫu và áp dụng quy trình dự đoán KNN.

Kết quả đầu ra là vector nhãn phân loại nhị phân.

---

### 3.1.6 Đánh giá độ chính xác

#### `accuracy(y_true, y_pred)`

Độ chính xác của mô hình được tính theo công thức:

```math
Accuracy = \frac{Number\ of\ Correct\ Predictions}{Total\ Predictions}
```

Chỉ số này phản ánh tỷ lệ dự đoán đúng trên toàn bộ dữ liệu kiểm tra.

---

### 3.1.7 Đặc điểm thuật toán

Ưu điểm:

- Dễ triển khai
- Không yêu cầu giai đoạn tối ưu phức tạp
- Hiệu quả với dữ liệu nhỏ

Hạn chế:

- Chi phí tính toán cao khi dữ liệu lớn
- Phụ thuộc mạnh vào quá trình chuẩn hóa dữ liệu
- Dễ bị ảnh hưởng bởi nhiễu

---

# 4. Thuật toán Decision Tree

## 4.1 Module `core/decision_tree.py`

### 4.1.1 Mục tiêu

Mô-đun triển khai mô hình Decision Tree phục vụ bài toán phân loại nhị phân.

Thuật toán hoạt động bằng cách phân tách dữ liệu theo các điều kiện ngưỡng nhằm tối đa hóa khả năng phân biệt giữa các lớp.

---

### 4.1.2 Cấu trúc nút cây

#### `class Node`

Mỗi nút trong cây quyết định bao gồm:

- `feature`: đặc trưng được sử dụng để phân tách
- `threshold`: giá trị ngưỡng
- `left`: nút con bên trái
- `right`: nút con bên phải
- `value`: nhãn dự đoán tại nút lá

---

### 4.1.3 Quy trình xây dựng cây

#### `fit(X, y)`

Hàm khởi tạo quá trình xây dựng cây bằng cách gọi đệ quy tới `_grow_tree()`.

---

#### `_grow_tree(X, y, depth)`

Hàm thực hiện xây dựng cây theo chiều sâu.

Quá trình phân tách sẽ dừng khi:

- đạt độ sâu tối đa (`max_depth`)
- toàn bộ mẫu thuộc cùng một lớp
- số lượng mẫu nhỏ hơn `min_samples_split`

Khi điều kiện dừng được thỏa mãn, hệ thống tạo nút lá với nhãn phổ biến nhất.

---

### 4.1.4 Lựa chọn phân tách tối ưu

#### `_best_split(X, y, feat_idxs)`

Hàm duyệt qua:

- các đặc trưng được chọn
- toàn bộ ngưỡng phân tách khả thi

để xác định phép phân tách tối ưu dựa trên chỉ số Information Gain.

---

### 4.1.5 Entropy và Information Gain

#### `_entropy(y)`

Entropy được sử dụng để đo mức độ hỗn loạn của dữ liệu:

```math
H = -\sum p_i \log_2(p_i)
```

Entropy càng nhỏ cho thấy dữ liệu tại nút càng đồng nhất.

---

#### `_information_gain(y, X_column, threshold)`

Information Gain được xác định bằng:

```math
IG = Entropy(parent) - Entropy(children)
```

Giá trị Information Gain càng lớn cho thấy phép phân tách càng hiệu quả.

---

### 4.1.6 Dự đoán dữ liệu

#### `predict(X)`

Quá trình suy luận được thực hiện bằng cách duyệt cây từ nút gốc đến nút lá.

Tại mỗi nút:

- nếu giá trị đặc trưng nhỏ hơn hoặc bằng `threshold`, dữ liệu đi sang nhánh trái
- ngược lại, dữ liệu đi sang nhánh phải

Kết quả cuối cùng được lấy tại nút lá.

---

### 4.1.7 Đặc điểm thuật toán

Ưu điểm:

- Dễ giải thích
- Không yêu cầu chuẩn hóa nghiêm ngặt
- Hoạt động tốt với dữ liệu phi tuyến

Hạn chế:

- Dễ overfitting
- Hiệu năng phụ thuộc vào cấu trúc cây
- Không ổn định khi dữ liệu thay đổi nhỏ

---

# 5. Thuật toán Logistic Regression

## 5.1 Module `core/logistic_regression.py`

### 5.1.1 Mục tiêu

Mô-đun triển khai mô hình Logistic Regression cho bài toán phân loại nhị phân.

Mô hình sử dụng:

- hàm sigmoid
- hàm mất mát Binary Cross Entropy
- Gradient Descent

để tối ưu tham số mô hình.

---

### 5.1.2 Cấu trúc mô hình

#### `class LogisticRegressionManual`

Các tham số chính:

- `learning_rate`
- `epochs`
- `weights`
- `bias`

Ngoài ra, hệ thống còn lưu:

- `loss_history`
- `accuracy_history`

nhằm phục vụ theo dõi quá trình huấn luyện.

---

### 5.1.3 Hàm sigmoid

#### `sigmoid(z)`

Hàm sigmoid chuyển đổi đầu ra tuyến tính thành xác suất:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

Giá trị đầu ra nằm trong khoảng `[0,1]`.

---

### 5.1.4 Hàm mất mát

#### `compute_loss(y_true, y_pred)`

Hệ thống sử dụng Binary Cross Entropy:

```math
L = -\frac{1}{n}\sum(y\log(p)+(1-y)\log(1-p))
```

Hàm mất mát phản ánh mức độ sai khác giữa:

- nhãn thực tế
- xác suất dự đoán

---

### 5.1.5 Quá trình huấn luyện

#### `fit(X, y)`

Quy trình huấn luyện bao gồm:

1. Khởi tạo trọng số và bias
2. Tính đầu ra tuyến tính
3. Áp dụng sigmoid
4. Tính gradient
5. Cập nhật tham số bằng Gradient Descent
6. Ghi nhận loss và accuracy theo từng epoch

Các công thức gradient:

```math
dw = \frac{1}{n}X^T(y_{pred} - y)
```

```math
db = \frac{1}{n}\sum(y_{pred} - y)
```

---

### 5.1.6 Quy trình dự đoán

#### `predict_probability(X)`

Trả về xác suất thuộc lớp dương.

---

#### `predict(X)`

Hệ thống sử dụng ngưỡng:

```text
p >= 0.5 → 1
p < 0.5 → 0
```

để chuyển xác suất thành nhãn phân loại.

---

### 5.1.7 Đặc điểm thuật toán

Ưu điểm:

- Huấn luyện nhanh
- Dễ diễn giải
- Phù hợp dữ liệu tuyến tính

Hạn chế:

- Khó xử lý quan hệ phi tuyến
- Phụ thuộc mạnh vào chất lượng đặc trưng

---

# 6. Quy trình huấn luyện mô hình

## 6.1 Module `core/trainer.py`

### 6.1.1 Mục tiêu

Mô-đun chịu trách nhiệm quản lý toàn bộ quá trình huấn luyện mô hình Logistic Regression bằng Mini-batch Gradient Descent.

---

### 6.1.2 Chia dữ liệu

#### `_train_test_split()`

Dữ liệu được chia thành:

- tập huấn luyện
- tập validation

theo tỷ lệ xác định.

Quá trình xáo trộn dữ liệu được thực hiện nhằm giảm sai lệch phân phối.

---

### 6.1.3 Mini-batch Gradient Descent

#### `_iterate_minibatches()`

Dữ liệu huấn luyện được chia thành nhiều batch nhỏ.

Mỗi batch được sử dụng để cập nhật gradient độc lập nhằm:

- tăng tốc huấn luyện
- giảm dao động gradient
- cải thiện khả năng hội tụ

---

### 6.1.4 Huấn luyện mô hình

#### `train_model_process()`

Quy trình huấn luyện gồm:

1. Chuẩn hóa dữ liệu
2. Chia tập train-validation
3. Khởi tạo mô hình
4. Huấn luyện theo epoch
5. Đánh giá trên validation set
6. Lưu lịch sử huấn luyện

Các chỉ số được ghi nhận:

- training loss
- validation loss
- training accuracy
- validation accuracy

---

# 7. Đánh giá mô hình

## 7.1 Module `core/evaluator.py`

### 7.1.1 Mục tiêu

Mô-đun cung cấp các chỉ số đánh giá hiệu năng của mô hình phân loại.

---

### 7.1.2 Các chỉ số đánh giá

#### `evaluate_model(y_true, y_pred)`

Hệ thống sử dụng các chỉ số:

- Accuracy
- Precision
- Recall
- F1-score

để đánh giá hiệu năng mô hình.

---

### 7.1.3 Ma trận nhầm lẫn

Mô hình sử dụng Confusion Matrix gồm:

- TP (True Positive)
- TN (True Negative)
- FP (False Positive)
- FN (False Negative)

Ma trận nhầm lẫn giúp phân tích chi tiết loại lỗi mà mô hình mắc phải.

---

### 7.1.4 Feature Importance

#### `get_feature_importance(data)`

Hàm xác định mức độ ảnh hưởng của từng đặc trưng thông qua hệ số tương quan với biến mục tiêu `Outcome`.

Các đặc trưng có hệ số tương quan lớn thường ảnh hưởng mạnh hơn đến kết quả dự đoán.

---

# 8. Pipeline dự đoán và so sánh mô hình

## 8.1 Module `core/predictor.py`

### 8.1.1 Mục tiêu

Mô-đun đóng vai trò điều phối toàn bộ pipeline dự đoán và so sánh hiệu năng giữa các mô hình học máy.

---

### 8.1.2 Quy trình xử lý

Pipeline hoạt động theo trình tự:

1. Chuẩn hóa dữ liệu
2. Chia tập huấn luyện và kiểm tra
3. Khởi tạo mô hình
4. Huấn luyện mô hình
5. Dự đoán dữ liệu kiểm tra
6. So sánh độ chính xác
7. Dự đoán dữ liệu đầu vào mới

---

### 8.1.3 Các mô hình được triển khai

Hệ thống hiện hỗ trợ:

- K-Nearest Neighbors
- Decision Tree
- Logistic Regression

---

### 8.1.4 Chuẩn hóa dữ liệu đầu vào

Dữ liệu đầu vào mới được chuẩn hóa bằng cùng thông số:

- mean
- standard deviation

được tính từ tập huấn luyện nhằm đảm bảo tính nhất quán giữa huấn luyện và suy luận.

---

# 9. Thuật ngữ chuyên môn

## 9.1 Thuật ngữ chính

| Thuật ngữ            | Mô tả                                  |
| -------------------- | -------------------------------------- |
| KNN                  | Thuật toán K-Nearest Neighbors         |
| Z-score              | Phương pháp chuẩn hóa dữ liệu          |
| Entropy              | Độ đo mức độ hỗn loạn dữ liệu          |
| Information Gain     | Lượng thông tin thu được sau phân tách |
| Sigmoid              | Hàm logistic chuyển đổi sang xác suất  |
| Binary Cross Entropy | Hàm mất mát cho phân loại nhị phân     |
| Gradient Descent     | Thuật toán tối ưu tham số              |
| Mini-batch           | Tập con dữ liệu dùng trong huấn luyện  |
| Confusion Matrix     | Ma trận đánh giá phân loại             |

---

## 9.2 Các viết tắt

| Viết tắt | Ý nghĩa              |
| -------- | -------------------- |
| TP       | True Positive        |
| TN       | True Negative        |
| FP       | False Positive       |
| FN       | False Negative       |
| BCE      | Binary Cross Entropy |

---

# 10. Kết luận

Hệ thống đã triển khai đầy đủ pipeline học máy cho bài toán dự đoán bệnh tiểu đường, bao gồm:

- tiền xử lý dữ liệu
- chuẩn hóa đặc trưng
- huấn luyện mô hình
- đánh giá hiệu năng
- suy luận dữ liệu mới

Ba thuật toán học máy chính được triển khai gồm:

- K-Nearest Neighbors
- Decision Tree
- Logistic Regression

Toàn bộ thuật toán được xây dựng thủ công nhằm phục vụ mục tiêu nghiên cứu, học thuật và phân tích cơ chế hoạt động của mô hình học máy.

Kiến trúc hệ thống được tổ chức theo hướng mô-đun hóa, giúp:

- dễ bảo trì
- dễ mở rộng
- thuận tiện cho việc thử nghiệm và so sánh thuật toán trong tương lai.
