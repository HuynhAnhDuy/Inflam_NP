import pandas as pd
import os
import glob

# 🔧 Thư mục chứa các file cần gộp
input_folder = "/home/andy/andy/Inflam_NP/visualization/InFlam_full_probs"

# 🔍 Lấy tất cả các file có hậu tố _prob.csv
pattern = os.path.join(input_folder, "*_prob.csv")
files = sorted(glob.glob(pattern))

print(f"🔍 Đang tìm thấy {len(files)} file để gộp...")

# 📥 Đọc và gộp tất cả file
df_list = []
for f in files:
    try:
        df = pd.read_csv(f, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding="latin1")
    df_list.append(df)

# 🧩 Gộp thành 1 DataFrame
df_full = pd.concat(df_list, ignore_index=True)

# 📤 Đặt tên file output (trong cùng thư mục với input)
output_file = os.path.join(input_folder, "InFlam_full_test_all_probs.csv")

# 💾 Xuất ra file CSV mới (utf-8)
df_full.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Đã gộp xong {len(files)} file.")
print(f"📁 Đã lưu file: {output_file}")
