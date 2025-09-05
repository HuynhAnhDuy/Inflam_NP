# -*- coding: utf-8 -*-
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# ===== CONFIG =====
BASE_PREFIX    = "Coconut_NP_similar_modified"
FINGERPRINTS   = ["ecfp","rdkit","maccs"]   # 3 FP -> 3 CSV
EPOCHS         = 30
BATCH_SIZE     = 32
RUN_SEEDS      = [42, 43, 44]               # -> y_prob_1..3
PRED_THRESHOLD = 0.5
# ==================

def build_model(input_dim: int):
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

def fit_and_predict_once(x_train, y_train, x_test, seed: int):
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
    x_train_r = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_r  = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    model = build_model(x_train.shape[1])
    model.fit(x_train_r, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=0.2, verbose=1)
    return model.predict(x_test_r, verbose=0).ravel()

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"Prob_{BASE_PREFIX}/Prob_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output folder: {out_dir}")

    for fp in FINGERPRINTS:
        fp_file = fp.lower()
        try:
            x_train_df = pd.read_csv(f"{BASE_PREFIX}_x_train_{fp_file}.csv", index_col=0)
            x_test_df  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fp_file}.csv", index_col=0)
            y_train    = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP {fp.upper()}] thiếu file: {e}")
            continue

        x_train = x_train_df.values
        x_test  = x_test_df.values
        ids     = x_test_df.index.astype(str).tolist()

        probs = []
        for i, seed in enumerate(RUN_SEEDS, start=1):
            print(f"\n🚀 {fp.upper()} | Run {i}/{len(RUN_SEEDS)} (seed={seed})")
            probs.append(fit_and_predict_once(x_train, y_train, x_test, seed))

        y_prob_1, y_prob_2, y_prob_3 = probs
        y_pro_average = (y_prob_1 + y_prob_2 + y_prob_3) / 3.0
        y_pred = (y_pro_average >= PRED_THRESHOLD).astype(int)

        out_df = pd.DataFrame({
            "ID": ids,
            "fingerprint": fp.upper(),
            "y_prob_1": y_prob_1,
            "y_prob_2": y_prob_2,
            "y_prob_3": y_prob_3,
            "y_pro_average": y_pro_average,
            "y_pred": y_pred
        })
        out_path = f"{out_dir}/{BASE_PREFIX}_test_pred_BiLSTM_{fp_file}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved ({fp.upper()}): {out_path} | rows={len(out_df)}")

if __name__ == "__main__":
    main()
