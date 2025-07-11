import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


# Importing Dataset

review = pd.read_csv('./Project/Reviews.csv')

# Part of speech tagging
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
        return wordnet.NOUN  # default to noun

# Stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Standardizing text
def standardize_text(text: str) -> list:
    # Convert text to lowercase
    text = text.lower()

    # Normalize unicode characters to ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize text
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

    return lemmatized

tqdm.pandas()
review['Processed Text'] = review['Text'].progress_apply(standardize_text)

review.to_csv('./Project/Reviews Pre-processed.csv')