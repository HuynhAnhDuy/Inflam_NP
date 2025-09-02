import numpy as np
import pandas as pd

# XGBoost
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("XGBoost chưa được cài. Cài bằng: pip install xgboost") from e

from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, precision_recall_curve,
    balanced_accuracy_score, 
)

def train_xgboost(
    x_train, x_test, y_train, y_test,
    n_estimators=500, max_depth=None, random_state=42,
    class_weight='balanced', n_jobs=-1,
    # Tham số XGBoost thường dùng
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=0.0, gamma=0.0, min_child_weight=1.0 
):
    # Chuẩn hóa input
    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    # max_depth: nếu None -> dùng default 6 của XGB
    if max_depth is None:
        max_depth = 6

    # Map class_weight='balanced' -> scale_pos_weight = #neg/#pos
    scale_pos_weight = None
    if class_weight == 'balanced':
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        if pos > 0:
            scale_pos_weight = float(neg) / float(pos)
        else:
            scale_pos_weight = 1.0

    # Tạo classifier (dùng 'hist' cho tốc độ; có thể đổi 'gpu_hist' nếu có GPU)
    params = dict(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        eval_metric="logloss",
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    clf = XGBClassifier(**params)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]

    # Metrics
    accuracy      = accuracy_score(y_test, y_pred)
    balanced_acc  = balanced_accuracy_score(y_test, y_pred)
    mcc           = matthews_corrcoef(y_test, y_pred)
    precision     = precision_score(y_test, y_pred, zero_division=0)
    recall        = recall_score(y_test, y_pred, zero_division=0)
    f1            = f1_score(y_test, y_pred, zero_division=0)

    # Specificity
    labels = np.unique(y_test)
    if set(labels) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR AUC
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_arr, prec_arr)

    return {
        "Accuracy Test": accuracy,
        "Balanced Accuracy Test": balanced_acc,
        "ROC AUC Test": roc_auc,
        "PR AUC Test": pr_auc,
        "MCC Test": mcc,
        "Precision Test": precision,
        "Sensitivity Test": recall,
        "Specificity Test": specificity,
        "F1 Test": f1
    }

def run_all_fingerprints(fingerprints, num_runs=3, base_prefix="3.InFlamNat"):
    results_all = {}

    for fp in fingerprints:
        print(f"\n=== Evaluating fingerprint: {fp.upper()} ===")

        # Đồng bộ hóa tên file theo lowercase để tránh lệch chữ hoa/thường
        fp_file = fp.lower()

        try:
            x_train = pd.read_csv(f"{base_prefix}_x_train_{fp_file}.csv", index_col=0).values
            x_test  = pd.read_csv(f"AISMPred_x_test_{fp_file}.csv", index_col=0).values
            y_train = pd.read_csv(f"{base_prefix}_y_train.csv", index_col=0).values.ravel()
            y_test  = pd.read_csv(f"AISMPred_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fp}: {e}")
            continue

        metrics_keys = [
            "Accuracy Test", "Balanced Accuracy Test", "ROC AUC Test", "PR AUC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run  # thay seed cho mỗi run
            metrics = train_xgboost(
                x_train, x_test, y_train, y_test,
                n_estimators=500, max_depth=None, random_state=seed,
                class_weight='balanced', n_jobs=-1
            )
            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

        # Trung bình ± SD (đối với NaN, dùng nanmean/nanstd)
        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fp] = summary

        print(f"--- {fp.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    return results_all

def main():
    # Đổi 'Estate' -> 'estate' cho chắc khớp tên file
    fingerprints = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
    results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)

    # Export to CSV (bảng hiển thị Mean ± SD dưới dạng text)
    df_export = pd.DataFrame({
        fp.upper(): {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for fp, metrics in results_by_fp.items()
    }).T
    df_export.to_csv("3.InFlamNat_XGB_fingerprint_metrics.csv")
    print("\nSaved results: 3.InFlamNat (XGB).")

if __name__ == "__main__":
    main()
