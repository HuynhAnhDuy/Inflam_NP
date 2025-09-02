import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve,
)
import warnings
import os
import random
from xgboost import XGBClassifier

# ================== CONFIG ==================
BASE_PREFIX = "1.Inflampred"
FINGERPRINTS = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
NUM_RUNS = 3
XGB_DEFAULTS = dict(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    gamma=0.0,
    min_child_weight=1.0,
)
# =============================================

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

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
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
    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "ROC AUC": _safe_auc_roc(y_true, y_prob),
        "PR AUC": _safe_auc_pr(y_true, y_prob),
        "MCC": mcc,
        "Precision": prec,
        "Sensitivity": rec,
        "Specificity": spec,
        "F1": f1
    }

# ===== TRAIN + EVAL CHO 1 SPLIT =====
def train_and_eval_once_xgb(X, y, seed=42, **xgb_params):
    np.random.seed(seed)
    random.seed(seed)

    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    max_depth_eff = 6 if (xgb_params["max_depth"] is None) else int(xgb_params["max_depth"])

    extra_params = {}
    if str(xgb_params["class_weight"]).lower() == "balanced":
        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        extra_params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=xgb_params["n_estimators"],
        max_depth=max_depth_eff,
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        reg_alpha=xgb_params["reg_alpha"],
        reg_lambda=xgb_params["reg_lambda"],
        gamma=xgb_params["gamma"],
        min_child_weight=xgb_params["min_child_weight"],
        n_jobs=xgb_params["n_jobs"],
        random_state=xgb_params["random_state"] if xgb_params["random_state"] else seed,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        **extra_params
    )

    model.fit(
        x_tr, y_tr,
        eval_set=[(x_te, y_te)],
        verbose=True,        
    )

    y_prob_train = model.predict_proba(x_tr)[:, 1]
    y_prob_test  = model.predict_proba(x_te)[:, 1]

    return compute_metrics(y_tr, y_prob_train), compute_metrics(y_te, y_prob_test)

# ===== CHẠY CÁC FINGERPRINTS =====
def run_all_fingerprints_xgb(fingerprints, num_runs=3, base_prefix=BASE_PREFIX, xgb_params=XGB_DEFAULTS):
    results_train, results_test = {}, {}
    for fp in fingerprints:
        print(f"\n=== 🔬 Fingerprint: {fp.upper()} ===")
        try:
            X = pd.read_csv(f"{base_prefix}_x_full_{fp.lower()}.csv", index_col=0).values
            y = pd.read_csv(f"{base_prefix}_y_full.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fp}: {e}")
            continue

        metric_names = ["Accuracy","Balanced Accuracy","ROC AUC","PR AUC","MCC","Precision","Sensitivity","Specificity","F1"]
        collected_train = {k: [] for k in metric_names}
        collected_test  = {k: [] for k in metric_names}

        for run in range(num_runs):
            seed = 42 + run
            print(f"  ▸ Run {run+1}/{num_runs} (seed={seed})")
            m_train, m_test = train_and_eval_once_xgb(X, y, seed=seed, **xgb_params)
            for k in metric_names:
                collected_train[k].append(m_train[k])
                collected_test[k].append(m_test[k])

        results_train[fp] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_train.items()}
        results_test[fp]  = {k: (np.nanmean(v), np.nanstd(v)) for k, v in collected_test.items()}
        print(f"  ✅ {fp.upper()} done.")
    return results_train, results_test

# ===== XUẤT 2 FILE: TRAIN & TEST =====
def export_two_files(results_train, results_test, base_prefix=BASE_PREFIX):
    def fmt(m,s): return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) and np.isfinite(s) else "nan"
    metric_order = ["Accuracy","Balanced Accuracy","ROC AUC","PR AUC","MCC","Precision","Sensitivity","Specificity","F1"]
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

    out_train = os.path.join("results",f"{base_prefix}_XGB_TRAIN_SUBSET.csv")
    out_test  = os.path.join("results",f"{base_prefix}_XGB_TEST_SUBSET.csv")
    df_train.to_csv(out_train, encoding="utf-8-sig")
    df_test.to_csv(out_test, encoding="utf-8-sig")
    
    print(f"\n📦 Saved results for prefix: {base_prefix} _ XGB_TRAIN/TEST SUBSETS")
    print(f"  - {out_train}")
    print(f"  - {out_test}")

# ===== MAIN =====
def main():
    warnings.filterwarnings("ignore")
    results_train, results_test = run_all_fingerprints_xgb(FINGERPRINTS, num_runs=NUM_RUNS, base_prefix=BASE_PREFIX, xgb_params=XGB_DEFAULTS)
    export_two_files(results_train, results_test, base_prefix=BASE_PREFIX)

if __name__ == "__main__":
    main()
