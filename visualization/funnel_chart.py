import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe

def plot_funnel_with_labels(steps, counts, colors=None, save_path="funnel_labels.svg", show_percent=False):
    """
    Vẽ funnel:
    - Hiển thị số liệu luôn.
    - % hiển thị tùy chọn qua show_percent (mặc định = True).
    - Tên bước bên ngoài, có leader line.
        """
    n = 5   
    steps = steps[:n]
    counts = counts[:n]

    base = counts[0]
    percentages = [c / base * 100 for c in counts]

    # Chiều rộng (tùy ý scale cho đẹp)
    widths = [w*2 for w in range(n*2, 0, -2)]  # [6,4,2] nếu n=3
    heights = [1.5] * n

    # Màu sắc mặc định
    if colors is None:
        colors = ["#0D1B2A", "#247B88", "#E0A800", "#CB8B2C","#8B0000"]


    fig, ax = plt.subplots(figsize=(9, 5))
    y_position = 0

    for i in range(n):
        top_width = widths[i]
        bottom_width = widths[i+1] if i < n-1 else 0
        y_top = y_position
        y_bottom = y_position - heights[i]
        y_center = (y_top + y_bottom) / 2

        # Vẽ tầng
        if i < n-1:
            coords = [
                (-top_width/2, y_top),
                (top_width/2, y_top),
                (bottom_width/2, y_bottom),
                (-bottom_width/2, y_bottom),
            ]
        else:  # cuối cùng = tam giác
            coords = [
                (-top_width/2, y_top),
                (top_width/2, y_top),
                (0, y_bottom),
            ]

        polygon = Polygon(coords, closed=True, facecolor=colors[i % len(colors)],
                          edgecolor="black", alpha=0.9)
        ax.add_patch(polygon)

        # Text hiển thị
        if show_percent and i > 0:
            ax.text(0, y_center+0.25, f"{counts[i]:,}",
                    va="center", ha="center", fontsize=11,
                    color="white", weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
            ax.text(0, y_center-0.25, f"({percentages[i]:.2f}%)",
                    va="center", ha="center", fontsize=9,
                    color="white", weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        else:
            ax.text(0, y_center, f"{counts[i]:,}",
                    va="center", ha="center", fontsize=11,
                    color="white", weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])

        # Nhãn + leader line
        label_x = max(widths)*0.7
        ax.text(label_x, y_center, steps[i],
                va="center", ha="left", fontsize=9, weight="bold", color="black")
        ax.plot([top_width/2, label_x-0.3], [y_center, y_center],
                color="gray", linewidth=1.0, linestyle="--")

        y_position = y_bottom

    # Giới hạn trục
    ax.set_xlim(-max(widths), max(widths)*1.2)
    ax.set_ylim(y_position - 0.5, 1)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, format="svg", transparent=True)


# ==== Demo ====
steps = steps = [
    "NPASS (Initial)",
    "LRo5 ≤1 violation & SA Score < 2",
    "Non-toxic Predictions\n(Dermal Toxicity and Carcinogenicity)",
    "Active NP Candidates\n(Predicted by Consensus Approach)",
    "Scaffold hopping",
]
counts = [93592, 2186, 584, 304, 20]

# Xuất 1 file SVG, bật/tắt % tùy ý
plot_funnel_with_labels(steps, counts, save_path="NP_funnel_labels_2.svg", show_percent=True)
