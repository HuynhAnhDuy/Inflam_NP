import pandas as pd

# === 1. Đọc 2 file SHAP ===
df_enrich = pd.read_csv("rgroup_shap_bilstm_enrichment_full.csv")
df_major = pd.read_csv("rgroup_shap_bilstm_majority_full.csv")

# === 2. Chỉ giữ R-group Positive (Active) ở cả 2 file ===
df_enrich_pos = df_enrich[df_enrich["effect"] == "Positive"]
df_major_pos = df_major[df_major["effect"] == "Positive"]

# === 3. Merge 2 file theo (rgroup_label, clean_smi) ===
df_common = pd.merge(
    df_enrich_pos[["rgroup_label", "clean_smi", "mean_shap"]],
    df_major_pos[["rgroup_label", "clean_smi", "mean_shap"]],
    on=["rgroup_label", "clean_smi"],
    suffixes=("_enrichment", "_majority"),
    how="inner"
)

# === 4. Tính mean_shap trung bình ===
df_common["mean_shap"] = df_common[["mean_shap_enrichment", "mean_shap_majority"]].mean(axis=1)

# === 5. Thêm cột effect = Positive ===
df_common["effect"] = "Positive"

# === 6. Chọn cột cần giữ ===
df_out = df_common[["rgroup_label", "clean_smi", "mean_shap", "effect"]]

# === 7. Xuất CSV ===
df_out.to_csv("rgroup_shap_bilstm_active.csv", index=False)

print(f"✅ Đã lưu {len(df_out)} R-group Active chung vào file")
