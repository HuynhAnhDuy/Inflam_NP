import pandas as pd
from pathlib import Path

# =========================
# 0) Đường dẫn các file
# =========================
SRC1_PATH = Path("scaffold_shap_summary_test.csv")        # file nguồn 1
SRC2_PATH = Path("scaffold_shap_summary_train.csv")       # file nguồn 2
SRC3_PATH = Path("scaffold_shap_summary_test_NP.csv")     # file nguồn 3
B_PATH    = Path("External_test_set_NP_external_with_scaffolds.csv")  # file B

OUT_PATH  = Path("External_scaffold_with_effect_and_source.csv")

# =========================
# 1) Hàm đọc + làm sạch chung
# =========================
def load_and_clean(path: Path) -> pd.DataFrame:
    # đọc, xử lý BOM ở header nếu có
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # chuẩn hóa cột scaffold nếu có
    if "scaffold" not in df.columns:
        raise ValueError(f"File {path} không có cột 'scaffold'")

    # ép kiểu string + strip + bỏ BOM/zero-width nếu có
    df["scaffold"] = (
        df["scaffold"]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u200b", "", regex=False)  # zero width space
        .str.strip()
    )

    return df

# =========================
# 2) Đọc & chuẩn hóa 3 file nguồn
# =========================
df1 = load_and_clean(SRC1_PATH)
df2 = load_and_clean(SRC2_PATH)
df3 = load_and_clean(SRC3_PATH)

# kiểm tra có cột effect không
for path, df in [(SRC1_PATH, df1), (SRC2_PATH, df2), (SRC3_PATH, df3)]:
    if "effect" not in df.columns:
        raise ValueError(f"File {path} không có cột 'effect'")

# thêm cột source_file để biết lấy từ file nào
df1["source_file"] = "scaffold_shap_summary_test"
df2["source_file"] = "scaffold_shap_summary_train"
df3["source_file"] = "scaffold_shap_summary_test_NP"

# chỉ giữ cột cần thiết
df1 = df1[["scaffold", "effect", "source_file"]]
df2 = df2[["scaffold", "effect", "source_file"]]
df3 = df3[["scaffold", "effect", "source_file"]]

# gộp 3 nguồn
df_src = pd.concat([df1, df2, df3], ignore_index=True)

# nếu 1 scaffold xuất hiện nhiều lần, giữ bản đầu tiên
df_src = df_src.drop_duplicates(subset="scaffold", keep="first")

print(f"Tổng số hàng trong 3 file nguồn sau khi gộp: {len(df_src)}")
print(f"Số scaffold khác nhau trong 3 file nguồn: {df_src['scaffold'].nunique()}")

# =========================
# 3) Đọc & chuẩn hóa file B
# =========================
dfB = load_and_clean(B_PATH)
print(f"Tổng số hàng trong file B: {len(dfB)}")
print(f"Số scaffold khác nhau trong file B: {dfB['scaffold'].nunique()}")

# =========================
# 4) Merge theo scaffold (left join để giữ nguyên file B)
# =========================
df_out = dfB.merge(df_src, on="scaffold", how="left", indicator=True)

# thống kê match
num_matched_rows = df_out["effect"].notna().sum()
num_matched_scaffolds = df_out.loc[df_out["effect"].notna(), "scaffold"].nunique()

print(f"Số hàng trong B có scaffold tìm thấy effect: {num_matched_rows}")
print(f"Số scaffold khác nhau trong B có match: {num_matched_scaffolds}")

# =========================
# 5) Lưu output
# =========================
df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Đã lưu file output tại: {OUT_PATH.resolve()}")
