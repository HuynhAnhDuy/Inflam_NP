import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc,
    precision_recall_curve
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

# ============== Train 1 fold ===========================
def train_one_fold(X_tr, y_tr, X_te, y_te, epochs=30, batch_size=32, seed=42):
    # reshape cho BiLSTM: [samples, timesteps=1, features]
    X_tr_r = X_tr.reshape((X_tr.shape[0], 1, X_tr.shape[1]))
    X_te_r = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))

    set_seed(seed)
    model = build_model(X_tr.shape[1])
    model.fit(
        X_tr_r, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # val nội bộ trong phần train của fold
        verbose=1
    )

    y_tr_prob = model.predict(X_tr_r, verbose=0).ravel()
    y_te_prob = model.predict(X_te_r, verbose=0).ravel()

    tr_m = compute_five_metrics(y_tr, y_tr_prob, threshold=0.5)
    te_m = compute_five_metrics(y_te, y_te_prob, threshold=0.5)
    return tr_m, te_m, y_tr_prob, y_te_prob

# ================== Run cho 1 fingerprint ==============
def run_cv_for_fingerprint(fp_name, y_sr):
    print(f"\n==============================")
    print(f" Running BiLSTM 5-fold CV for {fp_name.upper()} fingerprint")
    print(f"==============================")

    # ---- Load X cho fingerprint này
    X_df = pd.read_csv(f"InFlam_full_{fp_name}.csv", index_col=0)
    X = X_df.values.astype(np.float32)
    y = y_sr.values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    threshold = 0.5
    cols = ['Accuracy','MCC','AUROC','AUPRC','Sensitivity','Specificity']

    test_rows = []
    oof_probs = np.zeros_like(y, dtype=float)
    per_fold_store = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        seed_fold = 41 + fold
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        tr_m, te_m, y_tr_prob, y_te_prob = train_one_fold(
            X_tr, y_tr, X_te, y_te, epochs=30, batch_size=32, seed=seed_fold
        )
        test_rows.append([te_m[c] for c in cols])
        oof_probs[te_idx] = y_te_prob

        per_fold_store.append({
            'fold': fold,
            'te_idx': te_idx,
            'y_te_prob': y_te_prob,
            'te_metrics': te_m
        })

        print(f"Fold {fold} (seed={seed_fold}) "
              f"| TEST -> " + " | ".join([f"{k}:{te_m[k]:.3f}" if not np.isnan(te_m[k]) else f"{k}:NaN" for k in cols]))

    # ---- Lưu metrics + Average
    df_test  = pd.DataFrame(test_rows, columns=cols).astype(float)
    te_mean = df_test.mean().values
    te_sd   = df_test.std(ddof=0).values
    df_test_out  = df_test.round(3)
    df_test_out.loc['Average']  = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(te_mean, te_sd)]
    df_test_out.to_csv(f"bilstm_test_metrics_5CV_{fp_name}.csv",  index=False)
    print(f"\n📁 Saved metrics: bilstm_test_metrics_5CV_{fp_name}.csv")

    # ---- OOF predictions
    y_pred_oof = (oof_probs >= threshold).astype(int)
    pd.DataFrame({
        "y_true": y,
        "prob_avg": oof_probs,
        "y_predicted": y_pred_oof
    }).to_csv(f"bilstm_pred_oof_5CV_{fp_name}.csv", index=False)
    print(f"  - bilstm_pred_oof_5CV_{fp_name}.csv")

    # ---- Chọn best fold theo MCC
    best = sorted(
        per_fold_store,
        key=lambda d: (d['te_metrics']['MCC'], d['te_metrics']['AUROC'], d['te_metrics']['Accuracy']),
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
    }).to_csv(f"bilstm_pred_bestfold_5CV_{fp_name}.csv", index=False)

    print(f"\n🏅 Best fold ({fp_name}) = {best_fold} "
          f"(MCC={best['te_metrics']['MCC']:.3f}, "
          f"AUROC={best['te_metrics']['AUROC']:.3f}, "
          f"ACC={best['te_metrics']['Accuracy']:.3f})")
    print(f"  - bilstm_pred_bestfold_5CV_{fp_name}.csv")

# ======================== Main =========================
def main():
    y_sr = pd.read_csv("InFlam_full_y.csv", index_col=0).iloc[:, 0]
    for fp in ["ecfp", "rdkit", "maccs"]:
        run_cv_for_fingerprint(fp, y_sr)

if __name__ == "__main__":
    main()
