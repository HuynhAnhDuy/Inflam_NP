import csv

# tên file mặc định
INPUT_FILE = "NPASSv2.0.txt"
OUTPUT_FILE = "NPASSv2.0.csv"

# delimiter giả định: tab hoặc khoảng trắng; bạn chỉnh lại nếu cần
DELIMITER = None  # None = auto-split theo khoảng trắng, hoặc đặt ',' / '\t' / ';'

def iter_rows_text(fileobj, delimiter=None):
    if delimiter is None:
        # tách theo khoảng trắng
        for line in fileobj:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()
    else:
        reader = csv.reader(fileobj, delimiter=delimiter)
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].startswith("#"):
                continue
            yield row

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        for row in iter_rows_text(f, DELIMITER):
            writer.writerow(row)

if __name__ == "__main__":
    main()
