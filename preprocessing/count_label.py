import pandas as pd

# đọc file
df = pd.read_csv("/home/andy/andy/Inflam_NP/preprocessing/3.InFlamNat_modified.csv")

# đếm số lượng mỗi nhãn
counts = df["Label"].value_counts()

print("Số mẫu label = 1:", counts.get(1, 0))
print("Số mẫu label = 0:", counts.get(0, 0))
