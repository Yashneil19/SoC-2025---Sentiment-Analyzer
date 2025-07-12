import torch
import joblib
import numpy as np
import re
import unicodedata
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load trained components
model = joblib.load('xgboost_bert_sentiment_model.pkl')
label_map = joblib.load('sentiment_label_map.pkl')
vocabulary = joblib.load('vocabulary.pkl')
idf_vector = joblib.load('idf_vector.pkl')

# Load MiniLM model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def standardize_text(text: str) -> list:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lemmatized

# TF-IDF computation
def compute_tf(sentence, vocab):
    tf_vector = [0] * len(vocab)
    for word in sentence:
        if word in vocab:
            idx = vocab.index(word)
            tf_vector[idx] += 1
    word_count = len(sentence)
    return [count / word_count if word_count > 0 else 0 for count in tf_vector]

def compute_tfidf(tf_vector, idf_vector):
    return [tf * idf for tf, idf in zip(tf_vector, idf_vector)]

# Get combined feature vector
def get_combined_features(text):
    processed = standardize_text(text)
    
    # TF-IDF vector
    tf_vector = compute_tf(processed, vocabulary)
    tfidf_vector = compute_tfidf(tf_vector, idf_vector)

    # BERT vector (MiniLM CLS)
    bert_embedding = bert_model.encode([' '.join(processed)])[0]  # shape: (384,)

    # Concatenate
    combined = np.hstack([tfidf_vector, bert_embedding])  # shape: (len(vocab) + 384,)
    return combined.reshape(1, -1)

# Prediction loop
if __name__ == "__main__":
    while True:
        user_input = input("Enter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        try:
            features = get_combined_features(user_input)
            pred = model.predict(features)[0]
            print("Predicted Sentiment:", label_map[pred])
        except Exception as e:
            print("Error during prediction:", e)
