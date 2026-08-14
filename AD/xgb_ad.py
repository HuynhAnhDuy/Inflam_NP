from datetime import datetime
import os
import numpy as np
import pandas as pd

# XGBoost
try:
  from xgboost import XGBClassifier
except ImportError as e:
  raise SystemExit("XGBoost chưa được cài. Cài bằng: pip install xgboost") from e

from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors

# ---- Cấu hình: chỉ cần chỉnh 1 lần ở đây ----
BASE_PREFIX = "InFlam_full"


# === Huấn luyện model và áp dụng AD (kNN Euclidean) ===
def train_xgboost_with_ad(
    x_train,
    x_test,
    y_train,
    y_test,
    k_values=range(2, 11),
    n_estimators=500,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    min_child_weight=1,
):
  x_train = np.asarray(x_train)
  x_test = np.asarray(x_test)
  y_train = np.asarray(y_train).ravel()
  y_test = np.asarray(y_test).ravel()

  # 1. Fit mô hình XGBoost
  n_pos = np.sum(y_train == 1)
  n_neg = np.sum(y_train == 0)
  scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

  params = dict(
      objective="binary:logistic",
      n_estimators=n_estimators,
      max_depth=max_depth,
      learning_rate=learning_rate,
      subsample=subsample,
      colsample_bytree=colsample_bytree,
      reg_alpha=reg_alpha,
      reg_lambda=reg_lambda,
      gamma=gamma,
      min_child_weight=min_child_weight,
      random_state=random_state,
      n_jobs=n_jobs,
      tree_method="hist",
      eval_metric="logloss",
      scale_pos_weight=scale_pos_weight,
      use_label_encoder=False,
  )

  clf = XGBClassifier(**params)
  clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

  y_prob_test = clf.predict_proba(x_test)[:, 1]
  y_prob_train = clf.predict_proba(x_train)[:, 1]

  # 2. Xử lý Applicability Domain (kNN Euclidean distance) cho từng k từ 2 đến 10
  ad_results = {}

  for k in k_values:
    # Sử dụng kNN để tính khoảng cách Euclidean đến k láng giềng gần nhất
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(x_train)

    # Khoảng cách của tập train đến k láng giềng của chính nó (bỏ qua láng giềng đầu tiên là chính nó khoảng cách = 0)
    train_distances, _ = nn.kneighbors(x_train)
    mean_train_distances = np.mean(train_distances, axis=1)

    # Tính ngưỡng AD: Mean + Z * Std (thường dùng Z = 0.5 hoặc 1.0)
    ad_threshold = np.mean(mean_train_distances) + 0.5 * np.std(
        mean_train_distances
    )

    # Khoảng cách của tập test đến k láng giềng trong tập train
    test_distances, _ = nn.kneighbors(x_test)
    mean_test_distances = np.mean(test_distances, axis=1)

    # Lọc các mẫu test nằm trong miền AD (khoảng cách nhỏ hơn hoặc bằng ngưỡng)
    in_ad_mask = mean_test_distances <= ad_threshold

    if np.sum(in_ad_mask) == 0:
      continue  # Tránh lỗi nếu không có mẫu nào thỏa mãn

    y_test_filtered = y_test[in_ad_mask]
    y_prob_filtered = y_prob_test[in_ad_mask]
    y_pred_filtered = (y_prob_filtered >= 0.5).astype(int)

    # Tính toán các metrics sau khi lọc AD
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
    balanced_acc = balanced_accuracy_score(y_test_filtered, y_pred_filtered)
    mcc = matthews_corrcoef(y_test_filtered, y_pred_filtered)
    precision = precision_score(
        y_test_filtered, y_pred_filtered, zero_division=0
    )
    recall = recall_score(y_test_filtered, y_pred_filtered, zero_division=0)
    f1 = f1_score(y_test_filtered, y_pred_filtered, zero_division=0)

    labels = np.unique(y_test_filtered)
    if set(labels).issubset({0, 1}) and len(labels) == 2:
      tn, fp, fn, tp = confusion_matrix(
          y_test_filtered, y_pred_filtered, labels=[0, 1]
      ).ravel()
      specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
      specificity = np.nan

    try:
      fpr, tpr, _ = roc_curve(y_test_filtered, y_prob_filtered)
      roc_auc = auc(fpr, tpr)
    except Exception:
      roc_auc = np.nan

    try:
      prec_arr, rec_arr, _ = precision_recall_curve(
          y_test_filtered, y_prob_filtered
      )
      pr_auc = auc(rec_arr, prec_arr)
    except Exception:
      pr_auc = np.nan

    ad_results[k] = {
        "metrics": {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "AUROC": roc_auc,
            "AUPRC": pr_auc,
            "MCC": mcc,
            "Precision": precision,
            "Sensitivity": recall,
            "Specificity": specificity,
            "F1": f1,
        },
        "n_samples_kept": int(np.sum(in_ad_mask)),
        "coverage_percent": float(
            (np.sum(in_ad_mask) / len(y_test)) * 100
        ),
        "y_prob_filtered": y_prob_filtered,
        "y_test_filtered": y_test_filtered,
    }

  return {
      "y_prob_train": y_prob_train,
      "y_prob_test": y_prob_test,
      "y_train_true": y_train,
      "y_test_true": y_test,
      "ad_results": ad_results,
  }


# === Chạy qua tất cả fingerprint ===
def run_all_fingerprints(fingerprints, num_runs=3):
  all_metrics_raw = []

  timestamp = datetime.now().strftime("%Y-%m-%d")
  prob_folder = f"Prob_InFlam_full/Prob_{timestamp}_xgb_AD"
  os.makedirs(prob_folder, exist_ok=True)
  print(f"\n📁 Sẽ lưu kết quả AD vào thư mục: {prob_folder}")

  for fp in fingerprints:
    print(f"\n=== Evaluating fingerprint with AD: {fp.upper()} ===")
    fp_file = fp.lower()

    try:
      x_train = pd.read_csv(
          f"{BASE_PREFIX}_x_train_{fp_file}.csv", index_col=0
      ).values
      x_test = pd.read_csv(
          f"{BASE_PREFIX}_x_test_{fp_file}.csv", index_col=0
      ).values
      y_train = pd.read_csv(
          f"{BASE_PREFIX}_y_train.csv", index_col=0
      ).values.ravel()
      y_test = pd.read_csv(
          f"{BASE_PREFIX}_y_test.csv", index_col=0
      ).values.ravel()
    except FileNotFoundError as e:
      print(f"[SKIP] Thiếu file cho {fp.upper()}: {e}")
      continue

    for run in range(num_runs):
      seed = 42 + run
      result = train_xgboost_with_ad(
          x_train,
          x_test,
          y_train,
          y_test,
          k_values=range(2, 11),
          n_estimators=500,
          max_depth=6,
          random_state=seed,
          n_jobs=-1,
      )

      # Lưu kết quả cho từng giá trị k từ 2 đến 10
      for k, ad_res in result["ad_results"].items():
        metrics = ad_res["metrics"].copy()
        metrics["Fingerprint"] = fp.upper()
        metrics["k_neighbor"] = k
        metrics["Run"] = run + 1
        metrics["Seed"] = seed
        metrics["N_Samples_Kept"] = ad_res["n_samples_kept"]
        metrics["Coverage(%)"] = round(ad_res["coverage_percent"], 2)
        all_metrics_raw.append(metrics)

        # Lưu file xác suất dự đoán sau khi lọc AD
        filtered_df = pd.DataFrame({
            "y_true": ad_res["y_test_filtered"],
            "y_prob": ad_res["y_prob_filtered"],
        })
        filtered_path = (
            f"{prob_folder}/{BASE_PREFIX}_test_prob_{fp_file}_k{k}_run{run+1}.csv"
        )
        filtered_df.to_csv(filtered_path, index=False)

      print(
          f"💾 Đã hoàn tất và lưu các file AD (k=2->10) cho Run {run+1} của"
          f" {fp.upper()}"
      )

  # Xuất file tổng hợp raw metrics đầy đủ
  df_raw = pd.DataFrame(all_metrics_raw)
  output_csv = f"{BASE_PREFIX}_XGB_AD_k2_10_metrics_raw.csv"
  df_raw.to_csv(output_csv, index=False)
  print(f"\n✅ Đã lưu file tổng hợp raw kết quả AD tại: {output_csv}")


# === Hàm chính ===
def main():
  # Chỉ định 3 feature chính theo yêu cầu của bạn (có thể thêm estate, phychem nếu cần)
  fingerprints = ["rdkit", "ecfp", "maccs"]
  run_all_fingerprints(fingerprints, num_runs=3)


if __name__ == "__main__":
  main()