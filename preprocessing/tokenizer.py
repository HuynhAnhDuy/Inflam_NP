"""
Huynh Anh Duy
"""
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==== 1. Cài đặt tokenization ====
BRACKET_RGX = re.compile(r"(\[[^\[\]]+\])")

def canonicalize(smiles):
    """Chuyển về SMILES canonical để thống nhất định dạng."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

def tokenize_smiles(smiles):
    """Tách SMILES thành token: giữ nguyên cụm trong [ ], còn lại theo từng ký tự."""
    parts = re.split(BRACKET_RGX, smiles)
    tokens = []
    for part in filter(None, parts):
        if BRACKET_RGX.fullmatch(part):
            tokens.append(part)
        else:
            tokens.extend(list(part))
    return tokens

# ==== 2. Hàm xử lý tập dữ liệu ====
def process_smiles_file(filepath, tokenizer=None, max_len=100, fit_tokenizer=False):
    df = pd.read_csv(filepath)
    df["canonical"] = df["canonical_smiles"].map(canonicalize)
    df = df.dropna(subset=["canonical"])
    df["tokens"] = df["canonical"].map(tokenize_smiles)
    df["tokens_str"] = df["tokens"].map(lambda toks: " ".join(toks))

    if tokenizer is None:
        tokenizer = Tokenizer(char_level=False, lower=False, filters='', oov_token='[UNK]')

    if fit_tokenizer:
        tokenizer.fit_on_texts(df["tokens_str"])

    encoded = tokenizer.texts_to_sequences(df["tokens_str"])
    padded = pad_sequences(encoded, maxlen=max_len, padding='post', truncating='post')
    
    # Gộp kết quả vào DataFrame
    padded_df = pd.DataFrame(padded, columns=[f"tok_{i+1}" for i in range(max_len)])
    result = pd.concat([df.reset_index(drop=True), padded_df], axis=1)
    
    return result, tokenizer

# ==== 3. Gọi thực hiện ====
MAXLEN = 100

# a) Train
train_df, tokenizer = process_smiles_file(
    filepath="capsule_x_train.csv",
    tokenizer=None,
    max_len=MAXLEN,
    fit_tokenizer=True
)

# b) Test
test_df, _ = process_smiles_file(
    filepath="irac_iris_x_test.csv",
    tokenizer=tokenizer,
    max_len=MAXLEN,
    fit_tokenizer=False
)

# ==== 4. Lưu kết quả ====
train_df.to_csv("irac_iris_x_train_tokenized.csv", index=False)
test_df.to_csv("irac_iris_x_test_tokenized.csv", index=False)

np.save("X_train_token.npy", train_df[[f"tok_{i+1}" for i in range(MAXLEN)]].values)
np.save("X_test_token.npy",  test_df [[f"tok_{i+1}" for i in range(MAXLEN)]].values)

print("✅ Đã token hóa và padding xong:")
print("  • x_train_tokenized.csv")
print("  • x_test_tokenized.csv")
