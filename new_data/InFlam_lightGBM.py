from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cấu hình tên file đầu ra: chỉ cần chỉnh 1 chỗ
BASE_PREFIX = "new_data2"

# Kiểm tra thư viện LightGBM
try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise SystemExit(
        "LightGBM chưa được cài. Cài bằng: pip install lightgbm"
    ) from e

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


# === Huấn luyện và tính các chỉ số ===
def train_lightgbm(
    x_train,
    x_test,
    y_train,
    y_test,
    n_estimators=500,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
):
  x_train = np.asarray(x_train)
  x_test = np.asarray(x_test)
  y_train = np.asarray(y_train).ravel()
  y_test = np.asarray(y_test).ravel()

  if max_depth is None:
    max_depth = -1

  clf = LGBMClassifier(
      objective="binary",
      n_estimators=n_estimators,
      max_depth=max_depth,
      num_leaves=num_leaves,
      learning_rate=learning_rate,
      subsample=subsample,
      colsample_bytree=colsample_bytree,
      reg_alpha=reg_alpha,
      reg_lambda=reg_lambda,
      random_state=random_state,
      class_weight=class_weight,
      n_jobs=n_jobs,
  )
  clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)
  y_prob_test = clf.predict_proba(x_test)[:, 1]
  y_prob_train = clf.predict_proba(x_train)[:, 1]

  accuracy = accuracy_score(y_test, y_pred)
  balanced_acc = balanced_accuracy_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  precision = precision_score(y_test, y_pred, zero_division=0)
  recall = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)

  labels = np.unique(y_test)
  if set(labels) == {0, 1}:
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
  else:
    specificity = np.nan

  fpr, tpr, _ = roc_curve(y_test, y_prob_test)
  auroc = auc(fpr, tpr)

  prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_test)
  auprc = auc(rec_arr, prec_arr)

  return {
      "metrics": {
          "Accuracy Test": accuracy,
          "Balanced Accuracy Test": balanced_acc,
          "AUROC Test": auroc,
          "AUPRC Test": auprc,
          "MCC Test": mcc,
          "Precision Test": precision,
          "Sensitivity Test": recall,
          "Specificity Test": specificity,
          "F1 Test": f1,
      },
      "y_prob_train": y_prob_train,
      "y_prob_test": y_prob_test,
      "y_train_true": y_train,
      "y_test_true": y_test,
  }


# === Chạy qua tất cả fingerprint ===
def run_all_fingerprints(fingerprints, num_runs=3):
  results_all = {}
  all_metrics_raw = []

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  prob_folder = f"Prob_{BASE_PREFIX}/Prob_LGBM"
  os.makedirs(prob_folder, exist_ok=True)
  print(f"\n📁 Sẽ lưu y_prob vào thư mục: {prob_folder}")

  for fp in fingerprints:
    print(f"\n=== Evaluating fingerprint: {fp.upper()} ===")
    fp_file = fp.lower()

    try:
      x_train = pd.read_csv(
          f"{BASE_PREFIX}_x_train_{fp_file}.csv", index_col=0
      ).values
      x_test = pd.read_csv(
          f"{BASE_PREFIX}_x_test_{fp_file}.csv", index_col=0
      ).values
      y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values[
          :, 0
      ]
      y_test = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values[
          :, 0
      ]
    except FileNotFoundError as e:
      print(f"[SKIP] Thiếu file cho {fp.upper()}: {e}")
      continue

    # ĐÃ ĐỔNG BỘ SỬ DỤNG AUROC Test VÀ AUPRC Test
    metrics_keys = [
        "Accuracy Test",
        "Balanced Accuracy Test",
        "AUROC Test",
        "AUPRC Test",
        "MCC Test",
        "Precision Test",
        "Sensitivity Test",
        "Specificity Test",
        "F1 Test",
    ]
    metrics_summary = {k: [] for k in metrics_keys}

    for run in range(num_runs):
      seed = 42 + run
      result = train_lightgbm(
          x_train,
          x_test,
          y_train,
          y_test,
          n_estimators=500,
          max_depth=None,
          random_state=seed,
          class_weight="balanced",
          n_jobs=-1,
      )

      metrics = result["metrics"]
      for k in metrics_keys:
        metrics_summary[k].append(metrics[k])

      metrics["Fingerprint"] = fp.upper()
      metrics["Run"] = run + 1
      metrics["Seed"] = seed
      all_metrics_raw.append(metrics)

      # === Lưu y_prob train và test ===
      train_df = pd.DataFrame(
          {"y_true": result["y_train_true"], "y_prob": result["y_prob_train"]}
      )
      test_df = pd.DataFrame(
          {"y_true": result["y_test_true"], "y_prob": result["y_prob_test"]}
      )

      train_path = (
          f"{prob_folder}/{BASE_PREFIX}_train_prob_{fp_file}_run{run+1}.csv"
      )
      test_path = (
          f"{prob_folder}/{BASE_PREFIX}_test_prob_{fp_file}_run{run+1}.csv"
      )

      train_df.to_csv(train_path, index=False)
      test_df.to_csv(test_path, index=False)

      print(f"💾 Đã lưu: {train_path}, {test_path}")

    # Tính trung bình ± SD
    summary = {
        k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()
    }
    results_all[fp] = summary

    print(f"\n📊 --- {fp.upper()} Results (Mean ± SD over {num_runs} runs) ---")
    for k, (mean_val, std_val) in summary.items():
      print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

  # Xuất toàn bộ kết quả từng run
  df_raw = pd.DataFrame(all_metrics_raw)
  df_raw.to_csv(f"{BASE_PREFIX}_LGBM_fingerprint_metrics_raw.csv", index=False)
  print(f"\n✅ Saved raw results: {BASE_PREFIX}_LGBM_fingerprint_metrics_raw.csv")

  return results_all


# === Hàm chính ===
def main():
  fingerprints = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
  results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)

  # Xuất bảng tóm tắt (Mean ± SD)
  df_export = pd.DataFrame({
      fp.upper(): {
          metric: f"{mean:.3f} ± {std:.3f}"
          for metric, (mean, std) in metrics.items()
      }
      for fp, metrics in results_by_fp.items()
  }).T
  df_export.to_csv(f"{BASE_PREFIX}_LGBM_fingerprint_metrics.csv")
  print(f"\n✅ Saved summary: {BASE_PREFIX}_LGBM_fingerprint_metrics.csv")


if __name__ == "__main__":
  main()