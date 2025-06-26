import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('./IMDB Dataset/IMDB Dataset New.csv')

# Text Length Analysis
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

# Plots
sns.histplot(df['word_count'], bins=30, kde=True)
plt.title('Word Count Distribution')
plt.show()

# Most Common Words 
all_words = ' '.join(df['review']).split()
word_freq = Counter(all_words).most_common(10)
print("Top 10 words:", word_freq)

# N-gram Analysis
def get_ngrams(texts, n=2, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

print("Top Bigrams:", get_ngrams(df['review'], 2))
print("Top Trigrams:", get_ngrams(df['review'], 3))

# POS Tagging
pos_counts = Counter()
for text in df['review']:
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    pos_counts.update(tag for word, tag in tags)

pos_df = pd.DataFrame(pos_counts.items(), columns=['POS', 'Frequency'])
sns.barplot(data=pos_df.sort_values('Frequency', ascending=False).head(10), x='POS', y='Frequency')
plt.title('Top POS Tags')
plt.show()

# Named Entity Recognition (NER) 
def extract_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

all_ents = Counter()
for doc in df['review']:
    all_ents.update([label for text, label in extract_ner(doc)])

ner_df = pd.DataFrame(all_ents.items(), columns=['Entity', 'Count'])
sns.barplot(data=ner_df.sort_values('Count', ascending=False), x='Entity', y='Count')
plt.title('NER Entity Types')
plt.show()

df.to_csv('./IMDB Dataset/IMDB Dataset New2.csv')