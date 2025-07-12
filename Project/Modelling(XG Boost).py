import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from xgboost import XGBClassifier
import joblib

# Load data
review = pd.read_csv('./Project/Reviews_BERT_TF-IDF.csv')

# Parse both TF-IDF and BERT vectors
review["TFIDF_Vector"] = review["TFIDF_Vector"].apply(ast.literal_eval)
review["BERT_Vector"] = review["BERT_Vector"].apply(ast.literal_eval)

# Sentiment labels
review["Sentiment"] = review["Score"].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

# Downsample to balance
df_neg = review[review["Sentiment"] == 0]
df_neu = review[review["Sentiment"] == 1]
df_pos = review[review["Sentiment"] == 2]
min_class_size = min(len(df_neg), len(df_neu), len(df_pos))

df_balanced = pd.concat([
    resample(df_neg, n_samples=min_class_size, random_state=42),
    resample(df_neu, n_samples=min_class_size, random_state=42),
    resample(df_pos, n_samples=min_class_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Concatenate TF-IDF and BERT features
tfidf = np.vstack(df_balanced["TFIDF_Vector"].values)
bert = np.vstack(df_balanced["BERT_Vector"].values)
X = np.hstack([tfidf, bert])  # Concatenated features

# Labels
y = df_balanced["Sentiment"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

# Confusion Matrix
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Export the model and metadata 
joblib.dump(model, 'xgboost_bert_sentiment_model.pkl')
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
joblib.dump(label_map, 'sentiment_label_map.pkl')

print("Model and label map saved successfully.")