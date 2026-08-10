import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Danh sách 5 loại feature độc lập
feature_types = ["ecfp", "maccs", "phychem", "rdkit", "estate"]

# Đọc toàn bộ nhãn test gốc
y_test_df = pd.read_csv("InFlam_full_y_test.csv")

print("=== ĐANG TRÍCH XUẤT TẬP TEST TRONG MIỀN ÁP DỤNG (IN-AD) ==-\n")

for ft in feature_types:
    print(f"--- Đang xử lý Feature: [{ft.upper()}] ---")

    # Đọc dữ liệu Train và Test của feature hiện tại
    train_file = f"InFlam_full_x_train_{ft}.csv"
    test_file = f"InFlam_full_x_test_{ft}.csv"

    X_train_ft = pd.read_csv(train_file).values
    X_test_ft = pd.read_csv(test_file).values

    # Chuẩn hóa dữ liệu dựa trên Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ft)
    X_test_scaled = scaler.transform(X_test_ft)

    # 1. Tính khoảng cách Euclid ngắn nhất từ mỗi mẫu Test đến Train (NN distance)
    distances = pairwise_distances(
        X_test_scaled, X_train_scaled, metric="euclidean"
    )
    min_test_distances = np.min(distances, axis=1)

    # 2. Xác định ngưỡng AD từ tập Train (phân vị thứ 95 nội bộ train)
    train_internal_dist = pairwise_distances(
        X_train_scaled, X_train_scaled, metric="euclidean"
    )
    np.fill_diagonal(train_internal_dist, np.inf)
    min_train_internal = np.min(train_internal_dist, axis=1)
    ad_threshold = np.percentile(min_train_internal, 95)

    # 3. Lọc các chỉ số boolean cho mẫu nằm TRONG miền áp dụng (In-AD)
    in_ad_mask = min_test_distances <= ad_threshold

    # Lọc dữ liệu X_test và y_test tương ứng
    X_test_in_ad = X_test_ft[in_ad_mask]
    y_test_in_ad = y_test_df[in_ad_mask]

    # 4. Lưu ra các file CSV mới để đánh giá lại mô hình
    x_output_name = f"InFlam_full_x_test_{ft}_in_ad.csv"
    y_output_name = f"InFlam_full_y_test_{ft}_in_ad.csv"

    pd.DataFrame(X_test_in_ad).to_csv(x_output_name, index=False)
    y_test_in_ad.to_csv(y_output_name, index=False)

    print(f"  > Ngưỡng AD: {ad_threshold:.4f}")
    print(
        f"  > Số lượng mẫu Test gốc: {len(X_test_ft)} | Số mẫu giữ lại trong AD: {len(X_test_in_ad)} ({len(X_test_in_ad)/len(X_test_ft)*100:.2f}%)"
    )
    print(f"  > Đã xuất file: {x_output_name} và {y_output_name}")
    print("-" * 50)

print(
    "\nHoàn tất! Bạn có thể dùng các file *_in_ad.csv này để nạp vào mô hình dự đoán và tính lại metrics."
)