import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 0) File paths
# =========================
TRAIN_PATH = Path("InFlam_full_x_train.csv")
TEST_PATH  = Path("InFlam_full_x_test.csv")
EXT_PATH   = Path("External_test_set_NP_external.csv")

OUT_DIR = Path("modified_outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# 1) Helpers
# =========================
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    required = {"Label", "canonical_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["canonical_smiles"] = (
        df["canonical_smiles"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )

    before = len(df)
    df = df.dropna(subset=["Label", "canonical_smiles"]).copy()
    df["Label"] = df["Label"].astype(int)

    if "Index" in df.columns:
        df["Index"] = pd.to_numeric(df["Index"], errors="coerce")

    df = df.drop_duplicates(subset=["canonical_smiles", "Label"]).reset_index(drop=True)

    print(f"[{csv_path.name}] loaded: {before} -> cleaned: {len(df)}")
    return df


def remove_overlaps_by_smiles(df: pd.DataFrame, smiles_to_remove: set) -> pd.DataFrame:
    before = len(df)
    out = df[~df["canonical_smiles"].isin(smiles_to_remove)].copy().reset_index(drop=True)
    print(f"Removed overlaps: {before - len(out)} rows removed (kept {len(out)})")
    return out

# =========================
# 2) Load data
# =========================
x_train = load_and_clean(TRAIN_PATH)
x_test  = load_and_clean(TEST_PATH)
A_ext   = load_and_clean(EXT_PATH)

if not set(A_ext["Label"].unique()).issubset({1}):
    print("WARNING: External set contains labels other than 1.")

# Ensure Index
if "Index" not in A_ext.columns:
    A_ext.insert(0, "Index", np.arange(len(A_ext), dtype=int))
elif A_ext["Index"].isna().any():
    A_ext["Index"] = np.arange(len(A_ext), dtype=int)

# Ensure Compound name
if "Compound name" not in A_ext.columns:
    A_ext.insert(1, "Compound name", ["unknown"] * len(A_ext))

# =========================
# 3) Remove overlaps
# =========================
A_smiles = set(A_ext["canonical_smiles"].unique())

x_train_mod = remove_overlaps_by_smiles(x_train, A_smiles)
x_test_mod  = remove_overlaps_by_smiles(x_test,  A_smiles)

train_smiles_mod = set(x_train_mod["canonical_smiles"].unique())
overlap_train_test = train_smiles_mod.intersection(set(x_test_mod["canonical_smiles"].unique()))
if overlap_train_test:
    x_test_mod = remove_overlaps_by_smiles(x_test_mod, train_smiles_mod)

x_train_mod.to_csv(OUT_DIR / "InFlam_full_x_train_modified.csv", index=False)
x_test_mod.to_csv(OUT_DIR / "InFlam_full_x_test_modified.csv", index=False)

# =========================
# 4) Build test_new (NO SHUFFLE) with negatives = 3x positives
# =========================
neg_pool = x_test_mod[x_test_mod["Label"] == 0].copy()

RANDOM_SEED = 42
NEG_MULTIPLIER = 3
n_pos = len(A_ext)
n_neg = NEG_MULTIPLIER * n_pos

if len(neg_pool) < n_neg:
    raise ValueError(
        f"Not enough negative samples for {NEG_MULTIPLIER}x sampling. "
        f"Need {n_neg}, but neg_pool has {len(neg_pool)}."
    )

neg_sample = neg_pool.sample(n=n_neg, replace=False, random_state=RANDOM_SEED).copy()

# Prepare negative rows
neg_sample = neg_sample[["canonical_smiles", "Label"]].copy()

max_ext_index = int(A_ext["Index"].max())
neg_indices = np.arange(max_ext_index + 1, max_ext_index + 1 + len(neg_sample), dtype=int)
neg_names = [f"no_name_{i+1}" for i in range(len(neg_sample))]

neg_sample.insert(0, "Index", neg_indices)
neg_sample.insert(1, "Compound name", neg_names)

# Final output columns
A_ext_out = A_ext[["Index", "Compound name", "canonical_smiles", "Label"]].copy()
neg_out   = neg_sample[["Index", "Compound name", "canonical_smiles", "Label"]].copy()

test_new = pd.concat([A_ext_out, neg_out], ignore_index=True)

test_new_path = OUT_DIR / f"test_new_external_seed{RANDOM_SEED}.csv"
test_new.to_csv(test_new_path, index=False)

print(f"Saved: {test_new_path}")
print(f"test_new size = {len(test_new)} (positives={n_pos}, negatives={n_neg})")
