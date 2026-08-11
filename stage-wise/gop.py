# -*- coding: utf-8 -*-
import pandas as pd

# ========= CONFIG =========
FILE_X_STAGE   = "x.csv"
FILE_SKIN_TOX  = "test_skin_toxicity.csv"     # Tên file chứa skin_toxicity
FILE_CANCER    = "test_carcinogenicity.csv"   # Tên file chứa carcinogenicity
OUTPUT_FINAL   = "test_final_merged_mới.csv"
# ==========================

def map_skin_toxicity_note(val):
    if pd.isna(val):
        return "Unknown"
    # Chuẩn hóa chuỗi để so sánh chính xác (bỏ khoảng trắng thừa, chuyển về dạng chuẩn nếu cần)
    val_str = str(val).strip()
    if val_str == "Non-corrosive, Non-irritant, Non-sensitizer":
        return "Non-toxic"
    else:
        return "Toxic"

def map_carcinogenicity_note(val):
    if pd.isna(val):
        return "Unknown"
    val_str = str(val).strip().lower()
    if val_str == "non-carcinogen":
        return "Non-toxic"
    else:
        return "Toxic"

def main():
    # 1. Đọc các file CSV
    print("Đang đọc các file dữ liệu...")
    df_stage = pd.read_csv(FILE_X_STAGE)
    df_skin = pd.read_csv(FILE_SKIN_TOX)
    df_cancer = pd.read_csv(FILE_CANCER)
    
    n_initial = len(df_stage)
    print(f"Số lượng mẫu ban đầu trong {FILE_X_STAGE}: {n_initial}")

    # 2. Kiểm tra các cột cần thiết có tồn tại không
    if "canonical_smiles" not in df_stage.columns:
        raise ValueError(f"File {FILE_X_STAGE} thiếu cột 'canonical_smiles'")
    if "canonical_smiles" not in df_skin.columns or "skin_toxicity" not in df_skin.columns:
        raise ValueError(f"File {FILE_SKIN_TOX} thiếu cột 'canonical_smiles' hoặc 'skin_toxicity'")
    if "canonical_smiles" not in df_cancer.columns or "carcinogenicity" not in df_cancer.columns:
        raise ValueError(f"File {FILE_CANCER} thiếu cột 'canonical_smiles' hoặc 'carcinogenicity'")

    # 3. Gộp dữ liệu lần lượt bằng merge (left join giữ nguyên df_stage)
    df_merged = pd.merge(df_stage, df_skin[['canonical_smiles', 'skin_toxicity']], on='canonical_smiles', how='left')
    df_merged = pd.merge(df_merged, df_cancer[['canonical_smiles', 'carcinogenicity']], on='canonical_smiles', how='left')

    # 4. Thêm các cột ghi chú (note) bên cạnh
    df_merged['skin_toxicity_note'] = df_merged['skin_toxicity'].apply(map_skin_toxicity_note)
    df_merged['carcinogenicity_note'] = df_merged['carcinogenicity'].apply(map_carcinogenicity_note)

    # --- 5. Sắp xếp lại thứ tự cột để skin_toxicity_note nằm ngay cạnh skin_toxicity ---
    cols = list(df_merged.columns)
    # Định vị trí cột skin_toxicity và chèn skin_toxicity_note ngay sau nó
    skin_idx = cols.index('skin_toxicity')
    cols.remove('skin_toxicity_note')
    cols.insert(skin_idx + 1, 'skin_toxicity_note')
    
    # Định vị trí cột carcinogenicity và chèn carcinogenicity_note ngay sau nó (nếu muốn tương tự)
    cancer_idx = cols.index('carcinogenicity')
    cols.remove('carcinogenicity_note')
    cols.insert(cancer_idx + 1, 'carcinogenicity_note')
    
    df_merged = df_merged[cols]

    # 6. Xuất file kết quả cuối cùng
    df_merged.to_csv(OUTPUT_FINAL, index=False)
    
    print(f"\nGộp dữ liệu và sắp xếp cột thành công!")
    print(f"- Tổng số dòng sau khi gộp: {len(df_merged)}")
    print(f"- Số mẫu có thông tin skin_toxicity: {df_merged['skin_toxicity'].notna().sum()}")
    print(f"- Số mẫu có thông tin carcinogenicity: {df_merged['carcinogenicity'].notna().sum()}")
    print(f"- Đã lưu file kết quả tại: {OUTPUT_FINAL}")

if __name__ == "__main__":
    main()