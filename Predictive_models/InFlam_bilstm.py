import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

# ===== BUILD BiLSTM MODEL =====
def build_model(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(1, input_dim))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ===== TRAIN + EVALUATE MODEL =====
def evaluate_model(x_train, y_train, x_test, y_test, epochs=30, batch_size=32, run_id=1, seed=42):
    # Set seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Reshape for LSTM input
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    model = build_model(x_train.shape[2])
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    print(f"\n📉 Training loss/val_loss for Run {run_id}:")
    for epoch in range(epochs):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        print(f"  Epoch {epoch+1:02d}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

    # Predict
    y_pred_prob = model.predict(x_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # PR AUC
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec_arr, prec_arr)

    return {
        "Accuracy Test": acc,
        "Balanced Accuracy Test": balanced_acc,
        "ROC AUC Test": roc_auc,
        "PR AUC Test": pr_auc,
        "MCC Test": mcc,
        "Precision Test": prec,
        "Sensitivity Test": rec,
        "Specificity Test": specificity,
        "F1 Test": f1
    }

# ===== CHẠY QUA CÁC FINGERPRINTS =====
def run_all_fingerprints(fingerprints, num_runs=3, base_prefix="3.InFlamNat"):
    results_all = {}

    for fp in fingerprints:
        print(f"\n=== 🔬 Evaluating fingerprint: {fp.upper()} ===")

        fp_file = fp.lower()

        try:
            x_train = pd.read_csv(f"{base_prefix}_x_train_{fp_file}.csv", index_col=0).values
            x_test = pd.read_csv(f"{base_prefix}_x_test_{fp_file}.csv", index_col=0).values
            y_train = pd.read_csv(f"{base_prefix}_y_train.csv", index_col=0).values.ravel()
            y_test = pd.read_csv(f"{base_prefix}_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] ❌ Thiếu file cho {fp}: {e}")
            continue

        metrics_keys = [
            "Accuracy Test", "Balanced Accuracy Test", "ROC AUC Test", "PR AUC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run
            print(f"\n🚀 Run {run+1}/{num_runs} for {fp.upper()} (seed={seed})...")
            metrics = evaluate_model(
                x_train, y_train, x_test, y_test,
                epochs=30, batch_size=32, run_id=run+1, seed=seed
            )
            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

        # Trung bình ± SD
        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fp] = summary

        print(f"\n📊 --- {fp.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    return results_all

# ===== MAIN =====
def main():
    fingerprints = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
    results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)

    # Xuất ra bảng kết quả
    df_export = pd.DataFrame({
        fp.upper(): {
            metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()
        }
        for fp, metrics in results_by_fp.items()
    }).T

    df_export.to_csv("3.InFlamNat_BiLSTM_fingerprint_metrics.csv")
    print("\n✅ Saved results 3.InFlamNat (BiLSTM)")

if __name__ == "__main__":
    main()
