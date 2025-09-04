import pandas as pd
import numpy as np

def combine_features_ecfp_scaffold(
    ecfp_train_path="InFlam_full_x_train_ecfp.csv",
    ecfp_test_path="InFlam_full_x_test_ecfp.csv",
    scaffold_train_path="InFlam_full_x_train_scaffold.csv",
    scaffold_test_path="InFlam_full_x_test_scaffold.csv",
    output_train_path="InFlam_full_x_train_ecfp+scaffold.csv",
    output_test_path="InFlam_full_x_test_ecfp+scaffold.csv"
):
    # === Load ECFP & Scaffold features
    ecfp_train = pd.read_csv(ecfp_train_path, index_col=0)
    ecfp_test = pd.read_csv(ecfp_test_path, index_col=0)
    scaffold_train = pd.read_csv(scaffold_train_path, index_col=0)
    scaffold_test = pd.read_csv(scaffold_test_path, index_col=0)

    # === Kiểm tra index có khớp không
    if not ecfp_train.index.equals(scaffold_train.index):
        raise ValueError("❌ Train index không khớp giữa ECFP và Scaffold")
    if not ecfp_test.index.equals(scaffold_test.index):
        raise ValueError("❌ Test index không khớp giữa ECFP và Scaffold")

    # === Gộp đặc trưng theo chiều ngang
    X_train_combined = pd.concat([ecfp_train, scaffold_train], axis=1)
    X_test_combined = pd.concat([ecfp_test, scaffold_test], axis=1)

    # === Lưu file kết hợp
    X_train_combined.to_csv(output_train_path)
    X_test_combined.to_csv(output_test_path)

    print("✅ Đã tạo đặc trưng kết hợp:")
    print(f"- Train shape: {X_train_combined.shape}")
    print(f"- Test shape : {X_test_combined.shape}")
    print(f"💾 Đã lưu:\n  {output_train_path}\n  {output_test_path}")
if __name__ == "__main__":
    combine_features_ecfp_scaffold()

