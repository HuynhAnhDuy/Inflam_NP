import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    average_precision_score
)

def create_transformer(input_shape, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=(input_shape,))
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)  # key_dim giảm từ input_shape
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(input_shape)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

    x = GlobalAveragePooling1D()(out2)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_transformer_model(x_train, x_test, y_train, y_test, epochs=20, batch_size=32):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    model = create_transformer(input_shape=x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=0)

    y_test_pred_prob = model.predict(x_test).ravel()
    y_train_pred_prob = model.predict(x_train).ravel()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)

    # Metrics
    accuracy_test = accuracy_score(y_test, y_test_pred)
    balanced_acc_test = balanced_accuracy_score(y_test, y_test_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    tn_test, fp_test, _, _ = confusion_matrix(y_test, y_test_pred).ravel()
    specificity_test = tn_test / (tn_test + fp_test)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_prob)
    pr_auc_test = average_precision_score(y_test, y_test_pred_prob)

    return {
        "Accuracy Test": accuracy_test,
        "Balanced Accuracy Test": balanced_acc_test,
        "ROC AUC Test": roc_auc_test,
        "PR AUC Test": pr_auc_test,
        "MCC Test": mcc_test,
        "Precision Test": precision_test,
        "Sensitivity Test": recall_test,
        "Specificity Test": specificity_test,
        "F1 Test": f1_test
    }

def run_all_fingerprints(fingerprints, num_runs=3):
    results_all = {}

    for fp in fingerprints:
        print(f"\n=== Evaluating fingerprint: {fp.upper()} ===")
        
        # Load data
        x_train = pd.read_csv(f"1.Inflampred_x_train_{fp}.csv", index_col=0).values
        x_test = pd.read_csv(f"1.Inflampred_x_test_{fp}.csv", index_col=0).values
        y_train = pd.read_csv("1.Inflampred_y_train.csv", index_col=0).values.ravel()
        y_test = pd.read_csv("1.Inflampred_y_test.csv", index_col=0).values.ravel()

        metrics_summary = {metric: [] for metric in [
            "Accuracy Test", "Balanced Accuracy Test", "ROC AUC Test", "PR AUC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]}

        for run in range(num_runs):
            metrics = train_transformer_model(x_train, x_test, y_train, y_test)
            for key in metrics_summary:
                metrics_summary[key].append(metrics[key])
        
        # Tính trung bình ± SD
        summary = {metric: (np.mean(vals), np.std(vals)) for metric, vals in metrics_summary.items()}
        results_all[fp] = summary

        # In kết quả
        print(f"--- {fp.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for metric, (mean_val, std_val) in summary.items():
            print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

    return results_all

def main():
    fingerprints = ["ecfp", "estate","maccs", "phychem", "rdkit",]
    results_by_fp = run_all_fingerprints(fingerprints, num_runs=3)

    # Optional: Export to CSV
    df_export = pd.DataFrame({
        fp.upper(): {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for fp, metrics in results_by_fp.items()
    }).T
    df_export.to_csv("1.Inflampred_transformer_fingerprint_metrics.csv")
    print("\nSaved results to 'transformer_fingerprint_metrics.csv'.")

if __name__ == "__main__":
    main()
