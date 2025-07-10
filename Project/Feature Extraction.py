# Feature Extraction

import pandas as pd
import math
from transformers import BertTokenizer, BertModel

review = pd.read_csv('./Project/Reviews New.csv')

# Building the vocabulary from all sentences
vocabulary = set()
for sentence in review['Processed Text']:
    vocabulary.update(sentence)

# Convert to sorted list
vocabulary = sorted(list(vocabulary))

# Define BoW vector creation function
def create_bow_vector(sentence, vocab):
    vector = [0] * len(vocab)
    for word in sentence:
        if word in vocab:
            idx = vocab.index(word)
            vector[idx] += 1
    return vector

# Apply the function to each row
review['BoW_Vector'] = review['Processed Text'].apply(lambda x: create_bow_vector(x, vocabulary))


# TF-IDF

# TF (term frequency) vector creation
def compute_tf(sentence, vocab):
    tf_vector = [0] * len(vocab)
    for word in sentence:
        if word in vocab:
            idx = vocab.index(word)
            tf_vector[idx] += 1
    word_count = len(sentence)
    if word_count > 0:
        tf_vector = [count / word_count for count in tf_vector]
    return tf_vector

# Computing IDF (inverse document frequency)
def compute_idf(corpus, vocab):
    N = len(corpus)
    idf_vector = []
    for word in vocab:
        df = sum(1 for sentence in corpus if word in sentence)
        idf = math.log((N + 1) / (df + 1)) + 1  # Smoothed IDF
        idf_vector.append(idf)
    return idf_vector

# Computing TF-IDF vector
def compute_tfidf(tf_vector, idf_vector):
    return [tf * idf for tf, idf in zip(tf_vector, idf_vector)]

# Computing IDF
idf_vector = compute_idf(review['Processed Text'], vocabulary)

# Compute TF and then TF-IDF for each row
review['TF_Vector'] = review['Processed Text'].apply(lambda x: compute_tf(x, vocabulary))
review['TFIDF_Vector'] = review['TF_Vector'].apply(lambda tf: compute_tfidf(tf, idf_vector))



# Word Embeddings (BERT)

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Convert list of tokens to string 
review['Processed_Text_Str'] = review['Processed Text'].apply(lambda tokens: ' '.join(tokens))

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.squeeze().numpy()

review.to_csv('./Project/Reviews New2.csv')