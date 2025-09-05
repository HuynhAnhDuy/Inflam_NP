# -*- coding: utf-8 -*-
import os, random
import numpy as np
import pandas as pd
from datetime import datetime

# ===== CONFIG =====
BASE_PREFIX    = "Coconut_NP_similar_modified"     # tiền tố file
FINGERPRINTS   = ["ecfp","rdkit","maccs"]        # 3 FP -> 3 CSV
RUN_SEEDS      = [42, 43, 44]                    # số lần chạy để ensemble
PRED_THRESHOLD = 0.5

# XGBoost hyperparams
N_ESTIMATORS       = 500
MAX_DEPTH          = 6
LEARNING_RATE      = 0.05
SUBSAMPLE          = 0.8
COLSAMPLE_BYTREE   = 0.8
REG_ALPHA          = 0.0
REG_LAMBDA         = 0.0
GAMMA              = 0.0
MIN_CHILD_WEIGHT   = 1.0
CLASS_WEIGHT       = "balanced"                  # hoặc None
USE_GPU            = False                       # True -> 'gpu_hist', False -> 'hist'

# ==== XGBoost ====
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("XGBoost chưa được cài. Cài: pip install xgboost") from e


def build_xgb(random_state: int, y_train: np.ndarray):
    """Khởi tạo XGBClassifier với scale_pos_weight nếu cần."""
    params = dict(
        objective="binary:logistic",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        gamma=GAMMA,
        min_child_weight=MIN_CHILD_WEIGHT,
        random_state=random_state,
        n_jobs=-1,
        tree_method=("gpu_hist" if USE_GPU else "hist"),
        eval_metric="logloss",
        verbosity=1,
    )
    if CLASS_WEIGHT == "balanced":
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0
    return XGBClassifier(**params)


def fit_and_predict_once(x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, seed: int) -> np.ndarray:
    """Train 1 lần và trả về xác suất dương trên x_test."""
    random.seed(seed)
    np.random.seed(seed)
    clf = build_xgb(random_state=seed, y_train=y_train)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"Prob_{BASE_PREFIX}/Prob_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output folder: {out_dir}")

    for fp in FINGERPRINTS:
        fp_file = fp.lower()
        x_train_path = f"{BASE_PREFIX}_x_train_{fp_file}.csv"
        x_test_path  = f"{BASE_PREFIX}_x_test_{fp_file}.csv"
        y_train_path = f"{BASE_PREFIX}_y_train.csv"

        # ----- Load dữ liệu -----
        try:
            x_train_df = pd.read_csv(x_train_path, index_col=0)
            x_test_df  = pd.read_csv(x_test_path,  index_col=0)
            y_train    = pd.read_csv(y_train_path, index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP {fp.upper()}] Thiếu file: {e}")
            continue

        # ép kiểu & làm sạch nhanh
        x_train = x_train_df.values.astype(np.float32)
        x_test  = x_test_df.values.astype(np.float32)
        y_train = pd.Series(y_train).astype(np.int8).values

        # drop NaN nếu có
        if np.isnan(x_train).any() or np.isnan(x_test).any():
            print(f"[{fp.upper()}] ⚠️ Phát hiện NaN trong feature, sẽ thay bằng 0.0")
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
            x_test  = np.nan_to_num(x_test,  nan=0.0, posinf=0.0, neginf=0.0)

        ids = x_test_df.index.astype(str).tolist()
        print(f"[{fp.upper()}] x_train={x_train.shape}, x_test={x_test.shape}, y_train={y_train.shape}, pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}")

        # ----- Ensemble n lần theo RUN_SEEDS -----
        probs = []
        for i, seed in enumerate(RUN_SEEDS, start=1):
            print(f"\n🚀 {fp.upper()} | Run {i}/{len(RUN_SEEDS)} (seed={seed})")
            probs.append(fit_and_predict_once(x_train, y_train, x_test, seed))

        # ----- Tổng hợp & xuất -----
        probs = np.vstack(probs)           # shape = (n_runs, n_samples)
        y_pro_average = probs.mean(axis=0) # trung bình theo hàng
        y_pred = (y_pro_average >= PRED_THRESHOLD).astype(int)

        out_df = pd.DataFrame({
            "ID": ids,
            "fingerprint": fp.upper(),
            **{f"y_prob_{i+1}": probs[i] for i in range(probs.shape[0])},
            "y_pro_average": y_pro_average,
            "y_pred": y_pred
        })

        out_path = f"{out_dir}/{BASE_PREFIX}_test_pred_XGB_{fp_file}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved ({fp.upper()}): {out_path} | rows={len(out_df)}")

if __name__ == "__main__":
    main()
