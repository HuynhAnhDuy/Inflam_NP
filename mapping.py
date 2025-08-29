import pandas as pd
import sqlite3

# Kết nối vào file sqlite tải về
conn = sqlite3.connect("chembl_35.db")

# Truy vấn bảng cần thiết: molecule_dictionary + compound_structures
query = """
SELECT md.chembl_id, cs.canonical_smiles
FROM molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
"""
chembl_smiles = pd.read_sql_query(query, conn)
conn.close()

# Load file bạn đang có
df = pd.read_csv("InflamNat_Target_20210727.csv")
df["NP_ChEMBL_ID"] = df["NP_ChEMBL_ID"].astype(str).str.strip().str.upper()

# Merge để gán SMILES
merged = df.merge(chembl_smiles, left_on="NP_ChEMBL_ID", right_on="chembl_id", how="left")
merged = merged.drop(columns=["chembl_id"]).rename(columns={"canonical_smiles": "SMILES"})

# Xuất kết quả
merged.to_csv("InflamNat_Target_output.csv", index=False)
print("✅ Gán SMILES thành công từ ChEMBL local!")
