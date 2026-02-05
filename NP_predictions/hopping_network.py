import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Đọc dữ liệu
df = pd.read_csv("/home/andy/andy/Inflam_NP/NP_predictions/NPASS_common_scaffold_hopping_annotated_done.csv")

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
                label = f"P{counter_train}"
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

PARENT_COLOR = "#F8F8D9"  
CAND_COLOR   = "#E2D8F3"  
BORDER_COLOR = "#2B2B2B"   # dark charcoal (nhẹ hơn đen)
EDGE_COLOR   = "#1C1B1B"   # soft gray

plt.figure(figsize=(10,8))
nx.draw_networkx_edges(G, pos, alpha=0.6, width=edge_widths, edge_color=EDGE_COLOR)

nx.draw_networkx_nodes(
    G, pos, nodelist=train_nodes,
    node_color=PARENT_COLOR, edgecolors=BORDER_COLOR,
    linewidths=1, node_shape="o", node_size=800
)
nx.draw_networkx_nodes(
    G, pos, nodelist=cand_nodes,
    node_color=CAND_COLOR, edgecolors=BORDER_COLOR,
    linewidths=1, node_shape="h", node_size=800
)

nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")
plt.axis("off")

legend_elements = [
    mpatches.Patch(facecolor=PARENT_COLOR, edgecolor=BORDER_COLOR, label="Parent (P#)"),
    mpatches.Patch(facecolor=CAND_COLOR, edgecolor=BORDER_COLOR, label="Candidate (C#)")
]
plt.legend(handles=legend_elements, loc="best")

plt.rcParams['svg.fonttype'] = 'none'
out_svg = "scaffold_hopping_network.svg"
plt.savefig(out_svg, format="svg", bbox_inches="tight", transparent=True)
print(f"Đã lưu SVG tại: {out_svg}")
