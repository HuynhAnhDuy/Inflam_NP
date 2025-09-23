import svgutils.transform as sg
import math
import glob
import os
import re

def extract_num(filename):
    num = re.findall(r"(\d+)", os.path.basename(filename))
    return int(num[0]) if num else 0

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

        # scale và move
        root.scale(scale_factor)
        root.moveto(x, y)
        new_elems.append(root)

    fig.append(new_elems)
    fig.save(out_file)
    print(f"✅ Saved merged figure: {out_file} ({nrows} rows × {ncols} cols, {n} images)")

# --- Đọc và sort các file ---
input_folder = "/home/andy/andy/Inflam_NP/Scaffold_identify/shap_scaffold_split_XGB_20250923_122035"
svg_files = glob.glob(os.path.join(input_folder, "*.svg"))
svg_files = sorted(svg_files, key=extract_num)

# --- Chia thành 2 nhóm ---
part1 = svg_files[:8]   # 8 hình đầu
part2 = svg_files[8:]   # 8 hình sau

# --- Gộp từng nhóm ---
combine_svgs(part1, "Scaffold_merge_part1.svg", ncols=4, cell_size=(350,200), scale_factor=0.72)
combine_svgs(part2, "Scaffold_merge_part2.svg", ncols=4, cell_size=(350,200), scale_factor=0.72)
