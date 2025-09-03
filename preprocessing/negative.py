import requests
import pandas as pd
from collections import defaultdict

# ================== CONFIG ==================
# Ngưỡng cho các thước đo tính theo nM (IC50/EC50/Ki/Kd)
POTENCY_TYPES = {"IC50", "EC50", "Ki", "Kd"}
INHIBITION_TYPES = {"Inhibition", "% Inhibition", "%Inhibition"}  # phòng các biến thể tên
CUTOFF_NM = 10000.0   # > 10 µM coi là inactive
INHIBITION_MAX = 20.0 # %inhibition < 20% coi là inactive
ASSAY_CONFIDENCE_MIN = 7  # chỉ lấy assay có độ tin cậy cao nếu trường này tồn tại

# Ưu tiên Human; đặt True để chỉ lấy Human khi có (nếu trường target_organism/assay_organism tồn tại)
ONLY_HUMAN = True

# Danh sách target chống viêm (đã có trong code cũ + mở rộng)
# (Bạn có thể thêm/bớt tùy nhu cầu; càng nhiều target, càng dễ đạt 1270 negatives)
TARGETS = [
    # Đã dùng
    "CHEMBL230",   # COX-2 (PTGS2)
    "CHEMBL4637",  # COX-1 (PTGS1)
    "CHEMBL2835",  # p38 MAPK (MAPK14)
    "CHEMBL1824",  # TNF
    "CHEMBL1957",  # IL-6
    "CHEMBL3572",  # iNOS (NOS2)

    # Mở rộng (các mục tiêu thường gặp trong chống viêm / miễn dịch)
    # Nếu có target nào trả về ít dữ liệu, cứ để đó - không ảnh hưởng.
    "CHEMBL2147",  # 5-LOX (ALOX5) - thường có dữ liệu ức chế viêm
    "CHEMBL4005",  # 15-LOX (ALOX15)
    "CHEMBL260",   # JNK1 (MAPK8)
    "CHEMBL240",   # JNK2 (MAPK9)
    "CHEMBL2842",  # IKK-2 (IKBKB)
    "CHEMBL2971",  # p38 alpha/beta khác (check thêm nếu cần)
    "CHEMBL3880",  # PLA2G4A (cPLA2)
    "CHEMBL2970",  # ERK2 (MAPK1) - liên quan signal inflammation
    "CHEMBL2974",  # ERK1 (MAPK3)
    "CHEMBL4029",  # NF-κB essential modulator / liên quan pathway (tùy mức dữ liệu)
    # Bạn có thể thêm STAT3, JAKs, TLR4... nếu muốn mở rộng thêm
]

# ================== FETCH & FILTER ==================
def fetch_activities_for_target(tchembl_id, types_to_try):
    """Lấy toàn bộ activities cho 1 target với nhiều standard_type (phân trang)."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    results = []
    for stype in types_to_try:
        url = base_url
        params = {
            "target_chembl_id": tchembl_id,
            "standard_type": stype,
            "limit": 1000,
        }
        while True:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code != 200:
                # Có thể thử gọi lại 1-2 lần nếu cần; ở đây bỏ qua để đơn giản
                break
            data = r.json()
            acts = data.get("activities", [])
            results.extend(acts)
            next_url = data.get("next")
            if not next_url:
                break
            # Khi có next, ChEMBL trả URL đầy đủ; chuyển sang next và bỏ params
            url, params = next_url, None
    return results

def is_human(act):
    """Ưu tiên human; nếu không có trường, coi như hợp lệ."""
    if not ONLY_HUMAN:
        return True
    for k in ("target_organism", "assay_organism", "organism"):
        v = act.get(k)
        if v:
            return "Homo sapiens" in v or "Human" in v
    # Nếu không có thông tin organism, vẫn cho qua (để không quá khắt khe)
    return True

def good_confidence(act):
    """Giữ assay_confidence_score >= ngưỡng nếu có."""
    acs = act.get("assay_confidence_score")
    try:
        return (acs is None) or (int(acs) >= ASSAY_CONFIDENCE_MIN)
    except Exception:
        return True

def is_inactive(act):
    """Xác định inactive dựa trên loại thước đo."""
    stype = act.get("standard_type")
    sval = act.get("standard_value")
    sunits = act.get("standard_units")
    if not stype or sval is None:
        return False

    # Các thước đo tính theo nM
    if stype in POTENCY_TYPES:
        try:
            val = float(sval)
        except Exception:
            return False
        # Chỉ nhận nếu đơn vị tương thích (nM hoặc không ghi nhưng rõ ràng là nM)
        if sunits and sunits.lower() not in {"nm"}:
            # nếu đơn vị khác, bỏ qua để an toàn (có thể bổ sung chuyển đổi nếu cần)
            return False
        return val > CUTOFF_NM

    # %Inhibition: giá trị thấp coi là không ức chế
    if stype in INHIBITION_TYPES or (sunits and sunits.strip() == "%"):
        try:
            val = float(sval)
        except Exception:
            return False
        return val < INHIBITION_MAX

    return False

def collect_inactives():
    pool = []
    for t in TARGETS:
        print(f"Fetching: {t}")
        acts = fetch_activities_for_target(t, POTENCY_TYPES.union(INHIBITION_TYPES))
        for a in acts:
            if not is_human(a):
                continue
            if not good_confidence(a):
                continue
            if not is_inactive(a):
                continue
            smiles = a.get("canonical_smiles")
            mid = a.get("molecule_chembl_id")
            if not smiles or not mid:
                continue
            pool.append({
                "chembl_id": mid,
                "smiles": smiles,
                "target_id": t,
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "standard_relation": a.get("standard_relation"),
                "assay_confidence_score": a.get("assay_confidence_score"),
                "assay_organism": a.get("assay_organism"),
                "target_organism": a.get("target_organism"),
            })
    return pd.DataFrame(pool)

df = collect_inactives()

# ================== DEDUP & STATS ==================
# Deduplicate theo SMILES để tăng số hợp chất duy nhất
df_unique = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

print("Tổng bản ghi inactive (raw):", len(df))
print("Tổng hợp chất inactive (unique SMILES):", len(df_unique))

# ================== OUTPUT ==================
df_unique.to_csv("inactive_samples_expanded.csv", index=False)
print("Đã lưu: inactive_samples_expanded.csv")

# Gợi ý: nếu vẫn thiếu, có thể nới ngưỡng hoặc thêm target:
# - Nới: CUTOFF_NM = 5000, INHIBITION_MAX = 30
# - Thêm target vào TARGETS (JAK1/2/3, STAT3, TLR4, NLRP3, cGAS-STING, PDE4, HDAC6,...)
