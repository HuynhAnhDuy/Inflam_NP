import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Danh sách 5 loại feature độc lập
feature_types = ["ecfp", "maccs", "phychem", "rdkit", "estate"]

# Đọc toàn bộ nhãn test gốc
y_test_df = pd.read_csv("InFlam_full_y_test.csv")

ad_summary_results = []

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

    # Thống kê số lượng
    total_test = len(X_test_ft)
    retained_samples = len(X_test_in_ad)
    retained_percent = (retained_samples / total_test) * 100

    # 4. Lưu ra các file CSV mới để đánh giá lại mô hình
    x_output_name = f"InFlam_in_ad_x_test_{ft}.csv"
    y_output_name = f"InFlam_in_ad_y_test_{ft}.csv"

    pd.DataFrame(X_test_in_ad).to_csv(x_output_name, index=False)
    y_test_in_ad.to_csv(y_output_name, index=False)

    print(f"  > Ngưỡng AD: {ad_threshold:.4f}")
    print(
        f"  > Số lượng mẫu Test gốc: {total_test} | Số mẫu giữ lại trong AD: {retained_samples} ({retained_percent:.2f}%)"
    )
    print(f"  > Đã xuất file: {x_output_name} và {y_output_name}")
    print("-" * 50)

    # Thu thập kết quả cho bảng tổng hợp
    ad_summary_results.append(
        {
            "Feature": ft.upper(),
            "AD_Threshold": round(ad_threshold, 4),
            "Total_Test_Samples": total_test,
            "Retained_Samples": retained_samples,
            "Retained_Percent": round(retained_percent, 2),
        }
    )

# Lưu bảng tổng hợp kết quả AD ra file CSV chung
df_ad_summary = pd.DataFrame(ad_summary_results)
df_ad_summary.to_csv("applicability_domain_summary.csv", index=False)

print(
    "\nHoàn tất! Đã xuất các file dữ liệu In-AD và lưu bảng tổng hợp vào 'applicability_domain_summary.csv'."
)