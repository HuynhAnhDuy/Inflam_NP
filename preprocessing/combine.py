import pandas as pd

# Đọc 3 file dataset (đổi tên file cho đúng với bạn)
df1 = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/1.Inflampred_modified.csv")
df2 = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/2.AISMPred_modified.csv")
df3 = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/3.InFlamNat_preprocess.csv")

# Gộp tất cả lại với nhau theo hàng
merged = pd.concat([df1[['canonical_smiles','Label']], 
                    df2[['canonical_smiles','Label']], 
                    df3[['canonical_smiles','Label']]], 
                   ignore_index=True)

# Nếu muốn loại bỏ trùng lặp theo canonical_smiles + Label
merged = merged.drop_duplicates(subset=['canonical_smiles', 'Label'])

# Xuất ra file CSV
merged.to_csv("full.csv", index=False)
print("Đã tạo full.csv với", len(merged), "dòng")
