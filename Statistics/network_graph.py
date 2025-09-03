import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
file_path = "/home/andy/andy/Inflam_NP/Statistics/InFlam_full_dunn_Accuracy.csv"
df = pd.read_csv(file_path, index_col=0)
df = df.astype(float)

# Khởi tạo graph
G = nx.Graph()

# Thêm node
for model in df.index:
    G.add_node(model)

# Thêm edge nếu p < 0.05
for i in range(len(df)):
    for j in range(i+1, len(df)):
        p_val = df.iloc[i, j]
        if p_val < 0.05:
            G.add_edge(df.index[i], df.columns[j], weight=1 - p_val)  # Optional: weight = mức độ khác biệt

# Vẽ đồ thị
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)  # force-directed layout

nx.draw_networkx_nodes(G, pos, node_color="#0E03A3", node_size=5000)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color="white")
nx.draw_networkx_edges(G, pos, width=2, edge_color="#D3CEDF")

plt.title("Network Graph - Significant Dunn's Test Results (p < 0.05)", fontsize=14, weight='bold')
plt.axis("off")
plt.tight_layout()
plt.savefig("InFlam_Network_Accuracy.svg", format="svg", dpi=300)
