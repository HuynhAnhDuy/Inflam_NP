from itertools import combinations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

# === 1. Cấu hình đường dẫn ===
xgb_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_in_ad\Prob_2026-08-10_xgb"
bilstm_dir = r"D:\Andy\Inflam_NP\Predictive_models\Prob_InFlam_in_ad\Prob_2026-08-10_BiLSTM"

model_keys = [
    ("xgb", "rdkit"),
    ("xgb", "ecfp"),
    ("xgb", "maccs"),
    ("bilstm", "rdkit"),
    ("bilstm", "ecfp"),
    ("bilstm", "maccs"),
]

model_data = {}

print("--- ĐANG TẢI VÀ GỘP TRUNG BÌNH 3 RUNS CHO TỪNG MÔ HÌNH ---")
for algo, ft in model_keys:
  folder = xgb_dir if algo == "xgb" else bilstm_dir
  name = f"{algo.upper()}_{ft.upper()}"

  try:
    # Đường dẫn 3 file run
    path1 = os.path.join(folder, f"InFlam_in_ad_test_prob_{ft}_run1.csv")
    path2 = os.path.join(folder, f"InFlam_in_ad_test_prob_{ft}_run2.csv")
    path3 = os.path.join(folder, f"InFlam_in_ad_test_prob_{ft}_run3.csv")

    if not (os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3)):
      print(f"[THIẾU FILE CHO {name}] Bỏ qua.")
      continue

    # Đọc dữ liệu
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)

    # Kiểm tra cột
    if not all(col in df1.columns for col in ["y_true", "y_prob"]):
      print(f"[{name}] Lỗi: File không chứa đủ cột y_true, y_prob.")
      continue

    # Lấy nhãn thực tế từ run1 (giả định các run có thứ tự mẫu giống nhau tuyệt đối)
    y_true = df1["y_true"].values

    # Tính trung bình y_prob qua 3 runs
    mean_prob = (df1["y_prob"].values + df2["y_prob"].values + df3["y_prob"].values) / 3.0

    # Lưu vào DataFrame sạch
    df_model = pd.DataFrame({"y_true": y_true, "y_prob": mean_prob}).dropna()
    model_data[name] = df_model
    print(f"[{name}] Tải thành công! Số lượng mẫu: {len(df_model)}")

  except Exception as e:
    print(f"[LỖI XỬ LÝ {name}]: {e}")

print(f"\nTổng số mô hình tải thành công: {len(model_data)}/6")

if len(model_data) == 0:
  raise ValueError("Không tải được dữ liệu hợp lệ từ mô hình nào.")


# === 2. Hàm tính toán metrics ===
def evaluate_metrics(y_true, y_prob):
  y_pred = (y_prob >= 0.5).astype(int)

  fpr, tpr, _ = roc_curve(y_true, y_prob)
  auroc = auc(fpr, tpr)

  prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
  auprc = auc(rec_arr, prec_arr)

  mcc = matthews_corrcoef(y_true, y_pred)
  brier = brier_score_loss(y_true, y_prob)
  certainty = np.mean(2 * np.abs(y_prob - 0.5))

  return {
      "AUROC": round(auroc, 4),
      "AUPRC": round(auprc, 4),
      "MCC": round(mcc, 4),
      "Brier_Score": round(brier, 4),
      "Prediction_Certainty": round(certainty, 4),
  }


# === 3. Đánh giá mô hình đơn lẻ và tổ hợp Consensus ===
all_evaluations = []

print("\nĐang tính toán metrics cho Single Models...")
for name, df in model_data.items():
  res = evaluate_metrics(df["y_true"].values, df["y_prob"].values)
  res["Model_Type"] = "Single Model"
  res["Framework"] = name
  res["N_Samples"] = len(df)
  all_evaluations.append(res)

print("Đang tính toán metrics cho Consensus Models (Combo 2 -> 6)...")
model_names = list(model_data.keys())

for r in range(2, len(model_names) + 1):
  for combo in combinations(model_names, r):
    # Lấy DataFrame của các mô hình trong combo
    dfs_in_combo = [model_data[m] for m in combo]
    
    # Tìm số lượng mẫu nhỏ nhất hoặc ghép nối theo index dòng giả định
    # Vì mỗi mô hình có số lượng mẫu khác nhau do bộ lọc AD, ta lấy phần giao theo chiều dài tối thiểu 
    # hoặc gộp các cột y_prob lại và dropna() để chỉ giữ lại những dòng mà TẤT CẢ các mô hình trong combo đều có dự đoán
    
    probs_dict = {}
    for m in combo:
      # Reset index để đưa về chung một hệ trục vị trí dòng từ 0 đến N
      probs_dict[m] = model_data[m]["y_prob"].reset_index(drop=True)
    
    df_combo_probs = pd.DataFrame(probs_dict)
    
    # Lọc bỏ các dòng có giá trị NaN (tức là dòng nào mô hình này có mà mô hình kia không có thì loại bỏ để đảm bảo công bằng)
    df_combo_probs = df_combo_probs.dropna()
    
    n_samples = len(df_combo_probs)
    if n_samples == 0:
      continue

    # Lấy y_true tương ứng (lấy từ mô hình đầu tiên trong combo, sau khi đã dropna các vị trí tương ứng)
    # Để đảm bảo y_true khớp chính xác với các dòng không bị NaN của df_combo_probs:
    valid_indices = df_combo_probs.index
    y_true_combo = model_data[combo[0]]["y_true"].reset_index(drop=True).loc[valid_indices].values

    # Tính Consensus Probability (Trung bình cộng xác suất)
    consensus_probs = df_combo_probs.mean(axis=1).values

    res = evaluate_metrics(y_true_combo, consensus_probs)
    res["Model_Type"] = f"Combo {r}"
    res["Framework"] = " + ".join(combo)
    res["N_Samples"] = n_samples
    all_evaluations.append(res)

# === 4. Xuất kết quả ===
if len(all_evaluations) > 0:
  df_results = pd.DataFrame(all_evaluations)
  
  cols = ["Model_Type", "Framework", "N_Samples", "AUROC", "AUPRC", "MCC", "Brier_Score", "Prediction_Certainty"]
  df_results = df_results[[c for c in cols if c in df_results.columns]]

  output_csv = "consensus_ad_combos_evaluation_summary.csv"
  df_results.to_csv(output_csv, index=False)
  print(f"\n=== THÀNH CÔNG! Đã lưu kết quả tại: {output_csv} ===")
  print(df_results.sort_values(by="AUROC", ascending=False).head(10))
else:
  print("[CẢNH BÁO]: Không có kết quả nào được tạo ra.")