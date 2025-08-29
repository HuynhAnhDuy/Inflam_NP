import sqlite3

conn = sqlite3.connect("chembl_35.db")
cursor = conn.cursor()

# Kiểm tra cột trong bảng
cursor.execute("PRAGMA table_info(compound_structures)")
columns = cursor.fetchall()
for col in columns:
    print(col)

conn.close()
