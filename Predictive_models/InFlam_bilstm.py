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
from datetime import datetime

# ===== CHỈ CHỈNH 1 DÒNG NÀY =====
BASE_PREFIX = "InFlam_full"

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
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    model = build_model(x_train.shape[2])
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    print(f"\n📉 Training loss/val_loss for Run {run_id}:")
    for epoch in range(len(history.history['loss'])):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        print(f"  Epoch {epoch+1:02d}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

    y_test_prob = model.predict(x_test).ravel()
    y_test_pred = (y_test_prob > 0.5).astype(int)

    y_train_prob = model.predict(x_train).ravel()
    y_train_pred = (y_train_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(rec_arr, prec_arr)

    metrics = {
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

    return metrics, y_train_prob, y_test_prob, y_train, y_test

# ===== CHẠY QUA CÁC FINGERPRINTS =====
def run_all_fingerprints(fingerprints, num_runs=3):
    results_all = {}
    all_metrics_raw = []

    # === Tạo folder timestamp ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prob_folder = f"Prob_InFlam_full/Prob_{timestamp}"
    os.makedirs(prob_folder, exist_ok=True)
    print(f"\n📁 Sẽ lưu file xác suất tại: {prob_folder}")

    for fp in fingerprints:
        print(f"\n=== 🔬 Evaluating fingerprint: {fp.upper()} ===")
        fp_file = fp.lower()

        try:
            x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fp_file}.csv", index_col=0).values
            x_test = pd.read_csv(f"AISMPred_x_test_{fp_file}.csv", index_col=0).values
            y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
            y_test = pd.read_csv(f"AISMPred_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] ❌ Thiếu file cho {fp}: {e}")
            continue

        metrics_keys = [
            "Accuracy Test", "Balanced Accuracy Test", "AUROC Test", "AUPRC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run
            print(f"\n🚀 Run {run+1}/{num_runs} for {fp.upper()} (seed={seed})...")
            metrics, y_train_prob, y_test_prob, y_train_true, y_test_true = evaluate_model(
                x_train, y_train, x_test, y_test,
                epochs=30, batch_size=32, run_id=run+1, seed=seed
            )
            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

            metrics["Fingerprint"] = fp.upper()
            metrics["Run"] = run + 1
            metrics["Seed"] = seed
            all_metrics_raw.append(metrics)

            # === Save probability CSVs ===
            train_df = pd.DataFrame({
                'y_true': y_train_true,
                'y_prob': y_train_prob
            })
            test_df = pd.DataFrame({
                'y_true': y_test_true,
                'y_prob': y_test_prob
            })

            train_path = f"{prob_folder}/{BASE_PREFIX}_train_prob_{fp_file}_run{run+1}.csv"
            test_path = f"{prob_folder}/{BASE_PREFIX}_test_prob_{fp_file}_run{run+1}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"💾 Đã lưu: {train_path}, {test_path}")

        # Trung bình ± SD
        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fp] = summary

        print(f"\n📊 --- {fp.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    # Lưu kết quả từng run
    df_raw = pd.DataFrame(all_metrics_raw)
    df_raw.to_csv(f"{BASE_PREFIX}_BiLSTM_fingerprint_metrics_raw.csv", index=False)
    print(f"\n✅ Saved raw results: {BASE_PREFIX}_BiLSTM_fingerprint_metrics_raw.csv")

    return results_all

# ===== MAIN =====
def main():
    fingerprints = ["ecfp", "estate", "maccs", "phychem", "rdkit"]
    results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)

    # Xuất bảng kết quả tổng hợp
    df_export = pd.DataFrame({
        fp.upper(): {
            metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()
        }
        for fp, metrics in results_by_fp.items()
    }).T

    df_export.to_csv(f"{BASE_PREFIX}_BiLSTM_fingerprint_metrics.csv")
    print(f"\n✅ Saved summary: {BASE_PREFIX}_BiLSTM_fingerprint_metrics.csv")

if __name__ == "__main__":
    main()
