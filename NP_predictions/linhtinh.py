import svgutils.transform as sg
import math
import glob
import os
import re

def combine_svgs(svg_files, out_file, ncols=4, cell_size=(300,300), scale_factor=0.7):
    n = len(svg_files)
    nrows = math.ceil(n / ncols)

    # Kích thước figure tổng
    fig_width = ncols * cell_size[0]
    fig_height = nrows * cell_size[1]
    fig = sg.SVGFigure(str(fig_width), str(fig_height))

    new_elems = []
    for idx, svg in enumerate(svg_files):
        row, col = divmod(idx, ncols)
        x = col * cell_size[0]
        y = row * cell_size[1]

        fig_i = sg.fromfile(svg)
        root = fig_i.getroot()

        # scale để tránh chồng lấn
        root.scale(scale_factor)
        root.moveto(x, y)
        new_elems.append(root)

    fig.append(new_elems)
    fig.save(out_file)
    print(f"✅ Saved merged figure: {out_file} ({nrows} rows × {ncols} cols, {n} images)")

# --- Đọc tất cả file SVG trong folder ---
input_folder = "/home/andy/andy/Inflam_NP/NP_predictions/structures_output"
svg_files = glob.glob(os.path.join(input_folder, "C_*.svg"))

# Sắp xếp theo số trong tên file (T_1, T_2, ..., T_20)
def extract_num(fname):
    m = re.search(r"C_(\d+)\.svg", os.path.basename(fname))
    return int(m.group(1)) if m else 9999

svg_files = sorted(svg_files, key=extract_num)

# --- Chia thủ công: 8 + 8 + 4 ---
part1 = svg_files[:8]    # T1–T8
part2 = svg_files[8:16]  # T9–T16
part3 = svg_files[16:]   # T17–T20

combine_svgs(part1, "C_merged_part1.svg", ncols=4, cell_size=(350,200), scale_factor=0.72)
combine_svgs(part2, "C_merged_part2.svg", ncols=4, cell_size=(350,200), scale_factor=0.72)
combine_svgs(part3, "C_merged_part3.svg", ncols=4, cell_size=(300,200), scale_factor=0.6)
