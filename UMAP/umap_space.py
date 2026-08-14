import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import umap

# Danh sách 5 loại feature độc lập
feature_types = ["ecfp", "maccs", "phychem", "rdkit", "estate"]

# Đọc nhãn chung cho tập Train và Test (phục vụ bộ phân loại Partition Classifier)
y_train = pd.read_csv("InFlam_full_y_train.csv")
y_test = pd.read_csv("InFlam_full_y_test.csv")
y_train_arr = y_train["Label"].values
y_test_arr = y_test["Label"].values

results_summary = []

print("=== BẮT ĐẦU PHÂN TÍCH KHÔNG GIAN HÓA HỌC (TRAIN VS TEST) ==-\n")

for ft in feature_types:
    print(f"--- Đang xử lý Feature: [{ft.upper()}] ---")

    # Đọc file đặc trưng
    X_train_ft = pd.read_csv(f"InFlam_full_x_train_{ft}.csv").values
    X_test_ft = pd.read_csv(f"InFlam_full_x_test_{ft}.csv").values

    # Gộp dữ liệu và tạo nhãn partition
    all_data = np.vstack([X_train_ft, X_test_ft])
    partitions = ["Train"] * len(X_train_ft) + ["Test"] * len(X_test_ft)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    # Chạy UMAP giảm chiều về 2D
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    embedding = reducer.fit_transform(all_data_scaled)

    train_emb = embedding[: len(X_train_ft)]
    test_emb = embedding[len(X_train_ft) :]

    # ==========================================
    # VẼ VÀ LƯU BIỂU ĐỒ UMAP (TRAIN VS TEST)
    # ==========================================
    plt.figure(figsize=(6, 6))
    df_plot = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Partition": partitions,
        }
    )

    sns.scatterplot(
        data=df_plot,
        x="UMAP1",
        y="UMAP2",
        hue="Partition",
        palette={"Train": "#2117D8", "Test": "#CE3E40"},
        alpha=0.7,
        s=20,
        rasterized=True,
    )
    plt.xlabel(
        "UMAP 1", fontsize=12, fontweight="bold", fontstyle="italic"
    )
    plt.ylabel(
        "UMAP 2", fontsize=12, fontweight="bold", fontstyle="italic"
    )
    plt.legend(title="Partition",fontsize=12)
    plt.tight_layout()
    plt.savefig(f"chemical_space_{ft}_partitions.svg", format="svg", dpi=300)
    plt.close()

    # ==========================================
    # TÍNH TOÁN CÁC CHỈ SỐ ĐỊNH LƯỢNG
    # ==========================================
    train_centroid = np.mean(train_emb, axis=0)
    test_centroid = np.mean(test_emb, axis=0)
    centroid_dist = euclidean(train_centroid, test_centroid)

    w_dist_1 = wasserstein_distance(train_emb[:, 0], test_emb[:, 0])
    w_dist_2 = wasserstein_distance(train_emb[:, 1], test_emb[:, 1])

    X_clf = all_data_scaled
    y_clf = np.array([0] * len(X_train_ft) + [1] * len(X_test_ft))
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X_clf, y_clf, cv=5, scoring="roc_auc")
    mean_auroc = np.mean(cv_scores)

    # In kết quả console
    print(f"  > Centroid Distance: {centroid_dist:.4f}")
    print(
        f"  > Wasserstein Distance (UMAP1, UMAP2): {w_dist_1:.4f}, {w_dist_2:.4f}"
    )
    print(f"  > Partition Classifier AUROC: {mean_auroc:.3f}")
    print("-" * 50)

    results_summary.append(
        {
            "Feature": ft.upper(),
            "Centroid_Distance": round(centroid_dist, 4),
            "Wasserstein_UMAP1": round(w_dist_1, 4),
            "Wasserstein_UMAP2": round(w_dist_2, 4),
            "Partition_AUROC": round(mean_auroc, 3),
        }
    )

# Lưu bảng tổng hợp metrics ra file CSV
df_summary = pd.DataFrame(results_summary)
df_summary.to_csv("chemical_space_metrics_summary.csv", index=False)

print("\n=== HOÀN TẤT ===")
print("- Đã lưu 5 file hình UMAP Partition (.svg).")
print("- Đã lưu bảng tổng hợp vào 'chemical_space_metrics_summary.csv'.")