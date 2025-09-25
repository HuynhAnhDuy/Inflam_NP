import pandas as pd

# Đọc dữ liệu
x_train = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/InFlam_full_x_train - Copy.csv")
x_test = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/Inflampred_external_test_preprocess.csv")

# Tìm các canonical_smiles trùng nhau
overlap_smiles = set(x_train["canonical_smiles"]) & set(x_test["canonical_smiles"])

# Loại bỏ các dòng trong x_train có canonical_smiles trùng với x_test
x_train_modified = x_train[~x_train["canonical_smiles"].isin(overlap_smiles)]

# Xuất ra file mới
x_train_modified.to_csv("InFlam_modified_x_train.csv", index=False)

print(f"Số mẫu bị loại bỏ: {len(x_train) - len(x_train_modified)}")
print(f"x_train_modified còn lại: {len(x_train_modified)} dòng")
