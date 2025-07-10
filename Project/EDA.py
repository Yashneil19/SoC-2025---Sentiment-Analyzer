import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from wordcloud import WordCloud
from collections import Counter
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams
import spacy

nlp = spacy.load("en_core_web_sm")

review = pd.read_csv('./Project/Reviews New.csv')

# Handling Missing values
print("Number of missing values:",review['Text'].isnull().sum())
# NO missing values in the dataset


# Class distribution
print(review['Score'].value_counts())

sn.countplot(data=review, x='Score')
plt.title("Class Distribution")
plt.show()
# Dataset has more data to fit the model for positive reviews.(Less accuracy in case of negative/neutral reviews)

# Most Common Words 
all_words = ' '.join(review['Processed Text']).split()
word_freq = Counter(all_words).most_common(10)
print("Top 10 words:", word_freq)

# N-gram Analysis
def get_ngrams(texts, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

print("Top Bigrams:", get_ngrams(review['Processed Text'], 2))
print("Top Trigrams:", get_ngrams(review['Processed Text'], 3))

# POS Tagging
pos_counts = Counter()
for text in review['Text']:
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    pos_counts.update(tag for word, tag in tags)

pos_df = pd.DataFrame(pos_counts.items(), columns=['POS', 'Frequency'])
sn.barplot(data=pos_df.sort_values('Frequency', ascending=False).head(10), x='POS', y='Frequency')
plt.title('Top POS Tags')
plt.show()
