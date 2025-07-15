import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier
from tqdm import tqdm
import ast
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Config --------------------
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
EMBED_PATH = 'cls_embeddings.npy'
TMP_PATH = 'cls_embeddings.npy.tmp'
LABEL_PATH = 'cls_labels.npy'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('./Project/Reviews_BERT_TF-IDF.csv')

def join_tokens(tokens):
    return ' '.join(ast.literal_eval(tokens)) if isinstance(tokens, str) else ' '.join(tokens)

df['Processed_Text_Str'] = df['Processed Text'].apply(join_tokens)
df['Sentiment'] = df['Score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

# Balance dataset
df_neg = df[df['Sentiment'] == 0]
df_neu = df[df['Sentiment'] == 1]
df_pos = df[df['Sentiment'] == 2]
min_size = min(len(df_neg), len(df_neu), len(df_pos))

df_balanced = pd.concat([
    resample(df_neg, n_samples=min_size, random_state=42),
    resample(df_neu, n_samples=min_size, random_state=42),
    resample(df_pos, n_samples=min_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

texts = df_balanced['Processed_Text_Str'].tolist()
labels = df_balanced['Sentiment'].values
np.save(LABEL_PATH, labels, allow_pickle=False)  

# -------------------- Tokenizer & Model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Generate or Load CLS Embeddings (No Batching)
def generate_cls_embeddings_safe(texts, path, tmp_path): 

    total_len = len(texts)
    embedding_dim = 768

    print(f"Generating embeddings (no batching) and saving to {path}...")
    mmap = np.memmap(tmp_path, dtype='float32', mode='w+', shape=(total_len, embedding_dim))

    for idx, text in tqdm(enumerate(texts), total=total_len, desc="Generating Embeddings"):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        mmap[idx] = cls_vec
        mmap.flush()

    mmap.flush()
    del mmap
    os.rename(tmp_path, path)
    print(f" Saved embeddings to: {path}")

# Clean temp file if exists
if os.path.exists(TMP_PATH):
    os.remove(TMP_PATH)

generate_cls_embeddings_safe(texts, EMBED_PATH, TMP_PATH)

