import pandas as pd

# === 1. Load file CSV ===
df = pd.read_csv("/home/andy/andy/Inflam_NP/Scaffold_identify/3.InFlamNat_SHAP_with_scaffolds.csv")   # đổi tên file của bạn

# Kiểm tra cột
print("📂 Các cột có trong file:", df.columns.tolist())

# === 2. Scaffold cần lọc ===
target_scaffold = "O=c1c2ccccc2oc2ccccc12"   # thay scaffold bạn quan tâm

# === 3. Lọc dữ liệu ===
df_filtered = df[df["scaffold"] == target_scaffold].copy()

# === 4. Xuất kết quả ===
out_file = f"molecules_scaffold_{target_scaffold}.csv"
df_filtered.to_csv(out_file, index=False)

print(f"✅ Đã lọc {len(df_filtered)} molecules thuộc scaffold '{target_scaffold}'")
print(f"💾 Lưu file: {out_file}")
print(df_filtered.head())
