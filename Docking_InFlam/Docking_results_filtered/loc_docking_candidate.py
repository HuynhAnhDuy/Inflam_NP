import pandas as pd
import glob
import re

# 1. Hàm lấy target từ tên file
def extract_target(filename):
    match = re.search(r'^(.*?)_comparison_scores', filename)
    return match.group(1) if match else filename

# 2. Đọc và xử lý các file CSV
file_list = glob.glob("*_comparison_scores_*.csv")
all_data = []

for file in file_list:
    df = pd.read_csv(file)
    target = extract_target(file)
    
    # Lọc chỉ những hàng là 'potential'
    # Điều kiện: role là "Test" và affinity < -7
    potential_df = df[(df['role'] == 'Test') & (df['affinity (kcal/mol)'] < -7)].copy()
    
    if not potential_df.empty:
        potential_df['target_file'] = target
        # Chỉ giữ lại các cột cần thiết để tổng hợp
        all_data.append(potential_df[['ligand', 'affinity (kcal/mol)', 'target_file']])

# 3. Tổng hợp và xử lý
if all_data:
    combined_df = pd.concat(all_data)

    # Đếm số lần xuất hiện của mỗi ligand trên các target khác nhau
    ligand_counts = combined_df.groupby('ligand')['target_file'].nunique()
    valid_ligands = ligand_counts[ligand_counts >= 2].index

    # Lọc chỉ lấy các ligand xuất hiện >= 2 lần
    final_filtered = combined_df[combined_df['ligand'].isin(valid_ligands)]

    # 4. Tạo bảng tổng hợp (Pivot table)
    # Giá trị hiển thị là affinity (kcal/mol)
    summary_table = final_filtered.pivot_table(
        index='ligand', 
        columns='target_file', 
        values='affinity (kcal/mol)', 
        aggfunc='first' 
    )

    # Lưu kết quả
    summary_table.to_csv('potential_ligands_summary2.csv')
    print("Đã xuất file thành công: potential_ligands_affinity_summary.csv")
else:
    print("Không tìm thấy mẫu nào thỏa mãn điều kiện 'Test' và 'affinity < -7' trong các file.")