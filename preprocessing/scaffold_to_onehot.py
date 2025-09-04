import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import OneHotEncoder

# === Hàm tính scaffold ===
def compute_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    return None

# === Load dữ liệu X và Y, giữ index gốc ===
x_train = pd.read_csv("InFlam_full_x_train.csv", index_col=0)
y_train = pd.read_csv("InFlam_full_y_train.csv", index_col=0)
x_test = pd.read_csv("InFlam_full_x_test.csv", index_col=0)
y_test = pd.read_csv("InFlam_full_y_test.csv", index_col=0)

# === Tính scaffold cho train/test
x_train["scaffold"] = x_train["canonical_smiles"].apply(compute_scaffold)
x_test["scaffold"] = x_test["canonical_smiles"].apply(compute_scaffold)

# === Tạo mask để lọc những mẫu hợp lệ
mask_train = x_train["scaffold"].notnull()
mask_test = x_test["scaffold"].notnull()

x_train_clean = x_train[mask_train].copy()
y_train_clean = y_train.loc[mask_train].copy()

x_test_clean = x_test[mask_test].copy()
y_test_clean = y_test.loc[mask_test].copy()

# === Báo cáo số lượng mẫu ban đầu và sau xử lý ===
print("📊 SỐ LƯỢNG MẪU TRƯỚC & SAU XỬ LÝ:")
print(f"- Train trước xử lý: {len(x_train)}")
print(f"- Train sau xử lý  : {len(x_train_clean)}")
print(f"  -> Loại bỏ       : {len(x_train) - len(x_train_clean)} mẫu\n")

print(f"- Test trước xử lý : {len(x_test)}")
print(f"- Test sau xử lý   : {len(x_test_clean)}")
print(f"  -> Loại bỏ       : {len(x_test) - len(x_test_clean)} mẫu\n")

# === Encode scaffold dạng one-hot
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_vec = enc.fit_transform(x_train_clean[["scaffold"]])
X_test_vec = enc.transform(x_test_clean[["scaffold"]])

# === Chuyển thành DataFrame giữ lại index gốc
X_train_df = pd.DataFrame(X_train_vec, index=x_train_clean.index,
                          columns=enc.get_feature_names_out(["scaffold"]))
X_test_df = pd.DataFrame(X_test_vec, index=x_test_clean.index,
                         columns=enc.get_feature_names_out(["scaffold"]))

# === Lưu giữ nguyên index để khớp với file gốc
X_train_df.to_csv("InFlam_full_x_train_scaffold.csv")
X_test_df.to_csv("InFlam_full_x_test_scaffold.csv")
y_train_clean.to_csv("InFlam_full_y_train_clean.csv")
y_test_clean.to_csv("InFlam_full_y_test_clean.csv")

# === Kiểm tra lại số dòng khớp
print("🔎 KIỂM TRA KÍCH THƯỚC ĐẦU RA SAU XỬ LÝ:")
print(f"- x_train_scaffold: {X_train_df.shape}")
print(f"- y_train_clean   : {y_train_clean.shape}")
print(f"  -> Khớp         : {X_train_df.shape[0] == y_train_clean.shape[0]}")

print(f"- x_test_scaffold : {X_test_df.shape}")
print(f"- y_test_clean    : {y_test_clean.shape}")
print(f"  -> Khớp         : {X_test_df.shape[0] == y_test_clean.shape[0]}")
