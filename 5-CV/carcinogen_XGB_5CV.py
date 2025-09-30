import numpy as np 
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc,
    precision_recall_curve
)
import warnings
import random

# ==========================
# Reproducibility helpers
# ==========================
def set_seed(seed=None):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)

# ==========================
# 5 metrics chuẩn
# ===================== Metrics =========================
def compute_five_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = auc(fpr, tpr)

        # PRC
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(rec, prec)
    else:
        auroc = np.nan
        auprc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Accuracy': acc,
        'MCC': mcc,
        'AUROC': auroc,
        'AUPRC': auprc,       # ✅ thêm AUPRC
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

# ==========================
# Train 1 fold với XGB
# ==========================
def train_one_fold_xgb(X_tr, y_tr, X_te, seed=42):
    set_seed(seed)
    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

# ==========================
# Chạy CV cho 1 fingerprint
# ==========================
def run_cv_for_fingerprint(fp_name, y_sr):
    print(f"\n==============================")
    print(f" Running XGBoost 5-fold CV for {fp_name.upper()} fingerprint")
    print(f"==============================")

    # ---- Load X cho fingerprint này
    X_df = pd.read_csv(f"InFlam_full_{fp_name}.csv", index_col=0)
    X = X_df.values.astype(np.float32)
    y = y_sr.values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    threshold = 0.5
    cols = ['Accuracy','MCC','AUROC','AUPRC','Sensitivity','Specificity']

    test_rows = []
    fold_preds = []
    oof_probs = np.zeros_like(y, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        seed_fold = 41 + fold
        y_te_prob = train_one_fold_xgb(X_tr, y_tr, X_te, seed=seed_fold)

        te_metrics = compute_five_metrics(y_te, y_te_prob, threshold=threshold)
        test_rows.append([te_metrics[c] for c in cols])

        # lưu vào fold_preds
        fold_preds.append({
            "fold": fold,
            "te_idx": te_idx,
            "y_true": y_te,
            "y_prob": y_te_prob,
            "seed": seed_fold,
            "metrics": te_metrics
        })

        # ghi lại OOF prob
        oof_probs[te_idx] = y_te_prob

        msg = " | ".join([f"{k}:{te_metrics[k]:.3f}" if not np.isnan(te_metrics[k]) else f"{k}:NaN" for k in cols])
        print(f"Fold {fold} (seed={seed_fold}) TEST: {msg}")

    # ---- Metrics CSV
    df_test = pd.DataFrame(test_rows, columns=cols).astype(float)
    te_mean = df_test.mean().values
    te_sd   = df_test.std(ddof=0).values

    df_test_out = df_test.round(3)
    df_test_out.loc["Average"] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(te_mean, te_sd)]
    df_test_out.to_csv(f"xgb_test_metrics_5CV_{fp_name}.csv", index=False)

    print("\n📁 Saved metrics:")
    print(f"  - xgb_test_metrics_5CV_{fp_name}.csv")

    # ---- OOF predictions (cho toàn bộ dataset)
    y_pred_oof = (oof_probs >= threshold).astype(int)
    pd.DataFrame({
        "y_true": y,
        "prob": oof_probs,
        "y_predicted": y_pred_oof
    }).to_csv(f"xgb_pred_oof_5CV_{fp_name}.csv", index=False)
    print(f"  - xgb_pred_oof_5CV_{fp_name}.csv")

    # ---- Best fold theo MCC
    best = sorted(
        fold_preds,
        key=lambda d: (d["metrics"]["MCC"], d["metrics"]["AUROC"], d["metrics"]["Accuracy"]),
        reverse=True
    )[0]
    bfold, bseed = best["fold"], best["seed"]
    bm = best["metrics"]
    print(f"\n🏅 Best fold on TEST ({fp_name}): fold {bfold} "
          f"(MCC={bm['MCC']:.3f}, AUROC={bm['AUROC']:.3f}, ACC={bm['Accuracy']:.3f})")

    # ---- Xuất dự đoán TEST của best fold
    y_true_best = best["y_true"]
    y_prob_best = best["y_prob"]
    y_pred_best = (y_prob_best >= threshold).astype(int)

    pd.DataFrame({
        "y_true": y_true_best,
        "prob": y_prob_best,
        "y_predicted": y_pred_best
    }).to_csv(f"xgb_pred_bestfold_test_5CV_{fp_name}.csv", index=False)

    print("\n📁 Saved predictions:")
    print(f"  - xgb_pred_bestfold_test_5CV_{fp_name}.csv")


# ==========================
# Main
# ==========================
def main():
    # Load y (dùng chung cho cả 3 fingerprints)
    y_sr = pd.read_csv("InFlam_full_y.csv", index_col=0).iloc[:, 0]

    for fp in ["ecfp", "rdkit", "maccs"]:
        run_cv_for_fingerprint(fp, y_sr)

if __name__ == "__main__":
    main()
