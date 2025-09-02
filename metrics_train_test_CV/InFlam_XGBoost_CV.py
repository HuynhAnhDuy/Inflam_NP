import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import warnings
import random
import os

# ================== CONFIG ==================
BASE_PREFIX  = "1.Inflampred"      # prefix cho file dữ liệu vào/ra
FINGERPRINTS = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
N_SPLITS     = 5
SEED         = 42

# XGBoost hyperparams (CPU)
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=None,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric=["auc", "aucpr"],
    tree_method="hist",    # nhanh & ổn định trên CPU
    n_jobs=-1,
    random_state=SEED,
)
# ============================================


# ===== METRICS =====
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
    pr_auc  = _safe_auc_pr(y_true, y_prob)

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


# ===== BUILD XGB MODEL =====
def build_xgb(random_state=SEED, scale_pos_weight=None, **overrides):
    params = dict(XGB_PARAMS)
    params.update(overrides)
    params["random_state"] = random_state
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


# ===== TRAIN + EVAL 5-FOLD CV =====
def train_and_eval_cv_xgb(X, y, n_splits=5, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    metric_names = [
        "Accuracy", "Balanced Accuracy", "ROC AUC", "PR AUC",
        "MCC", "Precision", "Sensitivity", "Specificity", "F1"
    ]
    collected_train = {k: [] for k in metric_names}
    collected_val   = {k: [] for k in metric_names}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  ▸ Fold {fold_idx}/{n_splits}")

        x_tr, x_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # scale_pos_weight theo tỉ lệ lớp ở TRAIN-FOLD
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        spw = float(neg) / float(pos) if pos > 0 else 1.0

        model = build_xgb(random_state=seed + fold_idx, scale_pos_weight=spw)

        # Train (không early stopping, chạy đủ n_estimators)
        model.fit(x_tr, y_tr)

        # Predict prob
        y_prob_train = model.predict_proba(x_tr)[:, 1]
        y_prob_val   = model.predict_proba(x_va)[:, 1]

        m_train = compute_metrics(y_tr, y_prob_train)
        m_val   = compute_metrics(y_va, y_prob_val)

        for k in metric_names:
            collected_train[k].append(m_train[k])
            collected_val[k].append(m_val[k])

    return collected_train, collected_val


# ===== CHẠY NHIỀU FINGERPRINTS & GOM KẾT QUẢ =====
def run_all_fingerprints_xgb(fingerprints, n_splits=5, base_prefix=BASE_PREFIX, seed=SEED):
    results_mean_sd_train = {}
    results_mean_sd_val   = {}

    for fp in fingerprints:
        print(f"\n=== ✖️ XGBoost — Fingerprint: {fp.upper()} ===")
        fp_file = fp.lower()
        try:
            X = pd.read_csv(f"{base_prefix}_x_full_{fp_file}.csv", index_col=0).values
            y = pd.read_csv(f"{base_prefix}_y_full.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fp}: {e}")
            continue

        collected_train, collected_val = train_and_eval_cv_xgb(X, y, n_splits=n_splits, seed=seed)

        results_mean_sd_train[fp] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_train.items()}
        results_mean_sd_val[fp]   = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_val.items()}

        print(f"  ✅ {fp.upper()} done.")

    return results_mean_sd_train, results_mean_sd_val


# ===== XUẤT 2 FILE =====
def export_two_files(results_train, results_val, base_prefix=BASE_PREFIX):
    def fmt(m, s):
        return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) and np.isfinite(s) else "nan"

    metric_order = [
        "Accuracy", "Balanced Accuracy", "ROC AUC", "PR AUC",
        "MCC", "Precision", "Sensitivity", "Specificity", "F1"
    ]

    fps = [fp.upper() for fp in results_train.keys()]

    # ---- TRAIN (trên train-fold mỗi fold) ----
    df_train = pd.DataFrame(index=fps, columns=metric_order)
    for fp, mdict in results_train.items():
        for metric in metric_order:
            mean_val, sd_val = mdict[metric]
            df_train.loc[fp.upper(), metric] = fmt(mean_val, sd_val)

    # ---- VAL (trên fold giữ lại) ----
    df_val = pd.DataFrame(index=fps, columns=metric_order)
    for fp, mdict in results_val.items():
        for metric in metric_order:
            mean_val, sd_val = mdict[metric]
            df_val.loc[fp.upper(), metric] = fmt(mean_val, sd_val)
    # tạo thư mục nếu chưa có
    os.makedirs("results", exist_ok=True)

    out_train = os.path.join("results",f"{base_prefix}_XGB_5CV_TRAIN.csv")
    out_val   = os.path.join("results",f"{base_prefix}_XGB_5CV_VALID.csv")

    df_train.to_csv(out_train, encoding="utf-8-sig")
    df_val.to_csv(out_val,   encoding="utf-8-sig")

    print(f"\n📦 Saved results for prefix: {base_prefix} _ XGB_5-CV SUBSETS")
    print(f"  - {out_train}")
    print(f"  - {out_val}")


# ===== MAIN =====
def main():
    warnings.filterwarnings("ignore")

    results_train, results_val = run_all_fingerprints_xgb(
        fingerprints=FINGERPRINTS,
        n_splits=N_SPLITS,
        base_prefix=BASE_PREFIX,
        seed=SEED
    )

    export_two_files(results_train, results_val, base_prefix=BASE_PREFIX)

if __name__ == "__main__":
    main()
