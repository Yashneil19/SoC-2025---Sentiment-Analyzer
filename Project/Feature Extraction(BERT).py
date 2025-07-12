from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
import ast

review = pd.read_csv('./Project/Reviews Pre-processed.csv')

# Join tokenized text
def join_tokens(tokens):
    return ' '.join(ast.literal_eval(tokens)) if isinstance(tokens, str) else ' '.join(tokens)

review['Processed_Text_Str'] = review['Processed Text'].apply(join_tokens)

# 3-Class Sentiment Mapping: 0=Negative, 1=Neutral, 2=Positive
review['Sentiment'] = review['Score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

# Split into class-wise DataFrames
df_neg = review[review['Sentiment'] == 0]
df_neu = review[review['Sentiment'] == 1]
df_pos = review[review['Sentiment'] == 2]

# Downsample 
min_class_size = min(len(df_neg), len(df_neu), len(df_pos))
df_balanced = pd.concat([
    resample(df_neg, n_samples=min_class_size, random_state=42, replace=False),
    resample(df_neu, n_samples=min_class_size, random_state=42, replace=False),
    resample(df_pos, n_samples=min_class_size, random_state=42, replace=False),
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Extract texts 
texts = df_balanced['Processed_Text_Str'].tolist()

# Load MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode texts to BERT embeddings
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

df_balanced['BERT_Vector'] = embeddings.tolist()

df_balanced.to_csv('./Project/Reviews_BERT.csv')