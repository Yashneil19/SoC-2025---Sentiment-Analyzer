import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
EMBEDDING_DIM = 768
XGB_MODEL_PATH = 'xgb_cls_embeddings.pkl'
LABEL_MAP_PATH = 'label_map.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Components
print("Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("🔄 Loading trained XGBoost model...")
clf = joblib.load(XGB_MODEL_PATH)
label_map = joblib.load(LABEL_MAP_PATH)

# CLS Embedding Function
def get_cls_embedding(text: str):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

# Inference Function
def predict_sentiment(text: str):
    cls_vec = get_cls_embedding(text).reshape(1, -1)
    pred_label = clf.predict(cls_vec)[0]
    probas = clf.predict_proba(cls_vec)[0]
    return label_map[pred_label], dict(zip(label_map.values(), probas.round(4)))

# Run User Input Loop
if __name__ == "__main__":
    print("\nSentiment Predictor Ready!")
    print("Type your sentence below (or type 'exit' to quit):\n")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting sentiment predictor.")
            break

        if not user_input:
            print("Please enter a non-empty sentence.")
            continue

        label, probas = predict_sentiment(user_input)
        print(f"\nPredicted Sentiment: {label}")