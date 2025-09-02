import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
import random
import warnings
import os

# ================== CONFIG (chỉ cần đổi 1 chỗ) ==================
BASE_PREFIX = "1.Inflampred"      # prefix cho file dữ liệu vào/ra
FINGERPRINTS = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
NUM_RUNS = 3                     # số lần split ngẫu nhiên (80:20)
# --- RF hyperparams (có thể chỉnh) ---
N_ESTIMATORS = 500
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
N_JOBS = -1
# ================================================================

# ===== RF TRAINER (đúng bối cảnh bạn đang dùng) =====
def train_random_forest(
    x_train, x_test, y_train, y_test,
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42,
    class_weight=CLASS_WEIGHT, n_jobs=N_JOBS
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )
    rf.fit(x_train, y_train)

    # Xác suất (cho AUC/PR-AUC)
    y_prob_train = rf.predict_proba(x_train)[:, 1]
    y_prob_test  = rf.predict_proba(x_test)[:, 1]

    # Nhãn dự đoán (ngưỡng 0.5)
    y_pred_train = (y_prob_train > 0.5).astype(int)
    y_pred_test  = (y_prob_test  > 0.5).astype(int)

    return {
        "y_prob_train": y_prob_train,
        "y_prob_test":  y_prob_test,
        "y_pred_train": y_pred_train,
        "y_pred_test":  y_pred_test,
        "model": rf
    }

# ===== METRICS HELPERS =====
def _safe_auc_roc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return auc(fpr, tpr)

def _safe_auc_pr(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    return auc(rec_arr, prec_arr)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except ValueError:
        spec = np.nan

    roc_auc = _safe_auc_roc(y_true, y_prob)
    pr_auc = _safe_auc_pr(y_true, y_prob)

    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "MCC": mcc,
        "Precision": prec,
        "Sensitivity": rec,
        "Specificity": spec,
        "F1": f1
    }

# ===== TRAIN + EVAL CHO 1 LẦN SPLIT (80:20, stratify) =====
def train_and_eval_once(X, y, seed=42):
    # Split 80:20 (stratify)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Train RF (random_state = seed để mỗi split có hạt giống riêng)
    out = train_random_forest(
        x_tr, x_te, y_tr, y_te, random_state=seed
    )

    # Metrics Train/Test
    metrics_train = compute_metrics(y_tr, out["y_prob_train"])
    metrics_test  = compute_metrics(y_te, out["y_prob_test"])

    return metrics_train, metrics_test

# ===== CHẠY QUA CÁC FINGERPRINTS & GOM KẾT QUẢ =====
def run_all_fingerprints(fingerprints, num_runs=3, base_prefix=BASE_PREFIX):
    results_mean_sd_train = {}  # {fp: {metric: (mean, sd)}}
    results_mean_sd_test  = {}

    for fp in fingerprints:
        print(f"\n=== 🌲 RandomForest | Fingerprint: {fp.upper()} ===")
        fp_file = fp.lower()
        try:
            X = pd.read_csv(f"{base_prefix}_x_full_{fp_file}.csv", index_col=0).values
            y = pd.read_csv(f"{base_prefix}_y_full.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fp}: {e}")
            continue

        metric_names = [
            "Accuracy", "Balanced Accuracy", "ROC AUC", "PR AUC",
            "MCC", "Precision", "Sensitivity", "Specificity", "F1"
        ]
        collected_train = {k: [] for k in metric_names}
        collected_test  = {k: [] for k in metric_names}

        for run in range(num_runs):
            seed = 42 + run
            print(f"  ▸ Run {run+1}/{num_runs} (seed={seed})")
            m_train, m_test = train_and_eval_once(X, y, seed=seed)
            for k in metric_names:
                collected_train[k].append(m_train[k])
                collected_test[k].append(m_test[k])

        # Tính mean ± sd
        results_mean_sd_train[fp] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_train.items()}
        results_mean_sd_test[fp]  = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_test.items()}

        print(f"  ✅ {fp.upper()} done.")

    return results_mean_sd_train, results_mean_sd_test

# ===== XUẤT 2 FILE: TRAIN & TEST =====
def export_two_files(results_train, results_test, base_prefix=BASE_PREFIX):
    def fmt(m, s):
        return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) and np.isfinite(s) else "nan"

    metric_order = [
        "Accuracy", "Balanced Accuracy", "ROC AUC", "PR AUC",
        "MCC", "Precision", "Sensitivity", "Specificity", "F1"
    ]

    # Các fingerprint có kết quả (phòng khi thiếu file một số fp)
    fps = [fp.upper() for fp in results_train.keys()]

    # ---- TRAIN ----
    df_train = pd.DataFrame(index=fps, columns=metric_order)
    for fp, mdict in results_train.items():
        for metric in metric_order:
            mean_val, sd_val = mdict[metric]
            df_train.loc[fp.upper(), metric] = fmt(mean_val, sd_val)

    # ---- TEST ----
    df_test = pd.DataFrame(index=fps, columns=metric_order)
    for fp, mdict in results_test.items():
        for metric in metric_order:
            mean_val, sd_val = mdict[metric]
            df_test.loc[fp.upper(), metric] = fmt(mean_val, sd_val)

    # tạo thư mục nếu chưa có
    os.makedirs("results", exist_ok=True)

    # Tên file
    out_train = os.path.join("results",f"{base_prefix}_RF_TRAIN_SUBSET.csv")
    out_test  = os.path.join("results",f"{base_prefix}_RF_TEST_SUBSET.csv")

    # Lưu
    df_train.to_csv(out_train, encoding="utf-8-sig")
    df_test.to_csv(out_test,  encoding="utf-8-sig")

    print(f"\n📦 Saved results for prefix: {base_prefix} _ RF_TRAIN/TEST SUBSETS")
    print(f"  - {out_train}")
    print(f"  - {out_test}")

# ===== MAIN =====
def main():
    warnings.filterwarnings("ignore")

    results_train, results_test = run_all_fingerprints(
        fingerprints=FINGERPRINTS,
        num_runs=NUM_RUNS,
        base_prefix=BASE_PREFIX
    )

    export_two_files(results_train, results_test, base_prefix=BASE_PREFIX)

if __name__ == "__main__":
    main()
