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

df = pd.read_csv('./SoC-2025---Sentiment-Analyzer/Project/Reviews.csv')

# Handling Missing values
df['text'].isnull().sum()


df.to_csv('./SoC-2025---Sentiment-Analyzer/Project/Reviews New2.csv')