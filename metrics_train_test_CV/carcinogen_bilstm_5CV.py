import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import os

# =============== Reproducibility helpers ===============
def set_seed(seed=None):
    if seed is None:
        return
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ===================== Model ===========================
def build_model(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ===================== Metrics =========================
def compute_five_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = auc(fpr, tpr)
    else:
        auroc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': acc,
        'mcc': mcc,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

# ============== Train 1 fold ===========================
def train_one_fold(X_tr, y_tr, X_te, y_te, epochs=30, batch_size=32, seed=42):
    # reshape cho BiLSTM: [samples, timesteps=1, features]
    X_tr_r = X_tr.reshape((X_tr.shape[0], 1, X_tr.shape[1]))
    X_te_r = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))

    set_seed(seed)
    model = build_model(X_tr.shape[1])
    model.fit(
        X_tr_r, y_tr,
        epochs=30,
        batch_size=32,
        validation_split=0.2,  # val nội bộ trong phần train của fold
        verbose=0
    )

    y_tr_prob = model.predict(X_tr_r, verbose=0).ravel()
    y_te_prob = model.predict(X_te_r, verbose=0).ravel()

    tr_m = compute_five_metrics(y_tr, y_tr_prob, threshold=0.5)
    te_m = compute_five_metrics(y_te, y_te_prob, threshold=0.5)
    return tr_m, te_m, y_tr_prob, y_te_prob

# ======================== Main =========================
def main():
    # ---- Load data (giữ index) ----
    X_df = pd.read_csv("capsule_x_train_full_maccs.csv", index_col=0)
    y_sr = pd.read_csv("capsule_y_train_full.csv", index_col=0).iloc[:, 0]

    X = X_df.values.astype(np.float32)
    y = y_sr.values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    threshold = 0.5
    cols = ['accuracy','mcc','auroc','sensitivity','specificity']

    test_rows = []
    # OOF probabilities cho toàn bộ mẫu
    oof_probs = np.zeros_like(y, dtype=float)

    # Lưu thông tin từng fold để chọn "best fold" theo TEST metrics
    per_fold_store = []  # {'fold', 'te_idx', 'y_te_prob', 'te_metrics'}

    print("===== Pure 5-fold CV for BiLSTM (only TEST metrics saved) =====")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        seed_fold = 41 + fold  # 42..46
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        tr_m, te_m, y_tr_prob, y_te_prob = train_one_fold(
            X_tr, y_tr, X_te, y_te, epochs=30, batch_size=32, seed=seed_fold
        )
        test_rows.append([te_m[c] for c in cols])

        # OOF: ghi prob cho test fold này
        oof_probs[te_idx] = y_te_prob

        per_fold_store.append({
            'fold': fold,
            'te_idx': te_idx,
            'y_te_prob': y_te_prob,
            'te_metrics': te_m
        })

        print(f"Fold {fold} (seed={seed_fold}) "
              f"| TEST -> " + " | ".join([f"{k}:{te_m[k]:.3f}" if not np.isnan(te_m[k]) else f"{k}:NaN" for k in cols]))

    # ---- Chỉ lưu TEST metrics + Average (mean ± SD) ----
    df_test  = pd.DataFrame(test_rows, columns=cols).astype(float)
    te_mean = df_test.mean().values
    te_sd   = df_test.std(ddof=0).values
    df_test_out  = df_test.round(3)
    df_test_out.loc['Average']  = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(te_mean, te_sd)]
    df_test_out.to_csv("bilstm_val_metrics_5CV.csv",  index=False)
    print("\n📁 Saved metrics:")
    print("  - bilstm_val_metrics_5CV.csv")

    # ---- OOF predictions (khách quan trên toàn bộ dữ liệu) ----
    y_pred_oof = (oof_probs >= threshold).astype(int)
    pd.DataFrame({
        "y_true": y_sr.values,
        "prob_avg": oof_probs,
        "y_predicted": y_pred_oof
    }).to_csv("bilstm_pred_oof_5CV.csv", index=False)
    print("  - bilstm_pred_oof_5CV.csv")

    # ---- Chọn fold tốt nhất theo TEST MCC (tie-break: AUROC, Accuracy) ----
    best = sorted(
        per_fold_store,
        key=lambda d: (d['te_metrics']['mcc'], d['te_metrics']['auroc'], d['te_metrics']['accuracy']),
        reverse=True
    )[0]
    best_fold = best['fold']
    best_probs  = best['y_te_prob']
    best_y_true = y[best['te_idx']]
    best_y_pred = (best_probs >= threshold).astype(int)

    pd.DataFrame({
        "y_true": best_y_true,
        "prob_avg": best_probs,
        "y_predicted": best_y_pred
    }).to_csv("bilstm_pred_bestfold_5CV.csv", index=False)

    print(f"\n🏅 Best fold by TEST MCC: fold {best_fold} "
          f"(MCC={best['te_metrics']['mcc']:.3f}, "
          f"AUROC={best['te_metrics']['auroc']:.3f}, "
          f"ACC={best['te_metrics']['accuracy']:.3f})")
    print("  - bilstm_pred_bestfold.csv (ONLY the best fold's test set)")

if __name__ == "__main__":
    main()
