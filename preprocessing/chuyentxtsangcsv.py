import pandas as pd

# Đọc file txt (ở đây giả sử phân cách bằng tab)
df = pd.read_csv("NPASSv2.0.txt", sep="\t")

# Xuất sang CSV
df.to_csv("NPASSv2.0.csv", index=False)

print("Đã lưu thành công data.csv")
