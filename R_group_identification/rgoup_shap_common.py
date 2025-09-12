import pandas as pd

# === 1. Đọc file SHAP và file R-group khác ===
df_xgb = pd.read_csv("rgroup_shap_xgb_active.csv")
df_bilstm = pd.read_csv("rgroup_shap_bilstm_active.csv")
df_list = pd.read_csv("/home/andy/andy/Inflam_NP/R_group_identification/rgroup_only_active_clean.csv")

# === 2. Merge XGB và BiLSTM trước ===
df_merge = pd.merge(
    df_xgb[["rgroup_label", "clean_smi", "mean_shap", "effect"]],
    df_bilstm[["rgroup_label", "clean_smi", "mean_shap", "effect"]],
    on=["rgroup_label", "clean_smi"],
    how="inner",
    suffixes=("_xgb", "_bilstm")
)

# === 3. Merge tiếp với file list (giữ thêm group, rgroup_smiles, counts) ===
df_common = pd.merge(
    df_merge,
    df_list[["rgroup_label", "rgroup_smiles", "clean_smi", "active_count", "inactive_count"]],
    on=["rgroup_label", "clean_smi"],
    how="inner"
)

# === 4. Tính mean_shap trung bình ===
df_common["mean_shap_avg"] = df_common[["mean_shap_xgb", "mean_shap_bilstm"]].mean(axis=1)

# === 5. Gộp effect thành 1 cột duy nhất ===
def unify_effect(row):
    if row["effect_xgb"] == row["effect_bilstm"]:
        return row["effect_xgb"]
    else:
        return "Conflicted"

df_common["effect"] = df_common.apply(unify_effect, axis=1)

# === 6. Chọn cột cần giữ ===
df_out = df_common[[
    "rgroup_label", "rgroup_smiles", "clean_smi",
    "active_count", "inactive_count",
    "mean_shap_xgb", "mean_shap_bilstm", "mean_shap_avg", "effect"
]]

# === 7. Xuất ra CSV ===
output_file = "rgroup_shap_only_active.csv"
df_out.to_csv(output_file, index=False)

print(f"✅ Đã lưu {len(df_out)} R-group chung vào {output_file}")
