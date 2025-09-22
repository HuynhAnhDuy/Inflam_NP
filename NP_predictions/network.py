import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Đọc dữ liệu
df = pd.read_csv("/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping_annotated.csv")

# Tạo graph
G = nx.Graph()

node_id_map = {}   # map canonical_smiles -> số ID (T#/C#)
id_label_map = {}  # map ID -> smiles_name

counter_train = 1
counter_cand = 1

for _, row in df.iterrows():
    for smiles, name, ntype in [
        (row["canonical_smiles1"], row["smiles1_name"], "training"),
        (row["canonical_smiles2"], row["smiles2_name"], "candidate")
    ]:
        if smiles not in node_id_map:
            if ntype == "training":
                label = f"T{counter_train}"
                counter_train += 1
            else:
                label = f"C{counter_cand}"
                counter_cand += 1

            node_id_map[smiles] = label
            id_label_map[label] = name
            G.add_node(label, type=ntype, smiles=smiles)

    G.add_edge(node_id_map[row["canonical_smiles1"]],
               node_id_map[row["canonical_smiles2"]],
               weight=1.0)

# ==== Thêm cột ghi chú T#/C# vào file gốc ====
df["smiles1_ID"] = df["canonical_smiles1"].map(node_id_map)
df["smiles2_ID"] = df["canonical_smiles2"].map(node_id_map)

# Xuất file CSV mới
out_csv = "scaffold_hopping_results_with_ID.csv"
df.to_csv(out_csv, index=False)
print(f"Đã lưu file kết quả với ID tại: {out_csv}")

# ==== Vẽ network như cũ (tuỳ chọn) ====
pos = nx.spring_layout(G, k=2, seed=42)
edge_widths = [d.get('weight', 1.0) * 2 for (_, _, d) in G.edges(data=True)]

train_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "training"]
cand_nodes  = [n for n, d in G.nodes(data=True) if d["type"] == "candidate"]

plt.figure(figsize=(12,10))
nx.draw_networkx_edges(G, pos, alpha=0.7, width=edge_widths, edge_color="black")
nx.draw_networkx_nodes(G, pos, nodelist=train_nodes,
                       node_color="#DDF0A7", edgecolors="#101010",
                       linewidths=2, node_shape="o", node_size=800)
nx.draw_networkx_nodes(G, pos, nodelist=cand_nodes,
                       node_color="#69F0E7", edgecolors="#101010",
                       linewidths=2, node_shape="s", node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

plt.axis("off")
plt.title("Scaffold hopping network", fontsize=14,
          fontweight='bold', fontstyle='italic', family='sans-serif')

legend_elements = [
    mpatches.Patch(facecolor="#DDF0A7", edgecolor="black", label="Training (T#)"),
    mpatches.Patch(facecolor="#69F0E7", edgecolor="black", label="Candidate (C#)")
]
plt.legend(handles=legend_elements, loc="best")

plt.rcParams['svg.fonttype'] = 'none'
out_svg = "scaffold_hopping_network.svg"
plt.savefig(out_svg, format="svg", bbox_inches="tight", transparent=True)
print(f"Đã lưu SVG tại: {out_svg}")
