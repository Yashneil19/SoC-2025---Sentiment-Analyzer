import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

EMBED_PATH = 'cls_embeddings.npy' 
LABEL_PATH = 'cls_labels.npy'
NUM_SAMPLES = 127920  
EMBEDDING_DIM = 768

labels = np.load(LABEL_PATH)
print(f"Loading saved embeddings from {EMBED_PATH}")
cls_embeddings = np.memmap(EMBED_PATH, dtype='float32', mode='r', shape=(NUM_SAMPLES, EMBEDDING_DIM))

# Train
X_train, X_test, y_train, y_test = train_test_split(
    cls_embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Evaluation 
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)  # Probabilities for precision-recall curve

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

# Confusion Matrix
labels_map = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_map, yticklabels=labels_map)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Precision-Recall Curve
from itertools import cycle

n_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Binarize for PR curve

plt.figure(figsize=(8, 6))
colors = cycle(['navy', 'darkorange', 'teal'])

for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_probs[:, i])
    plt.plot(recall, precision, color=color, lw=2,
             label=f'{labels_map[i]} (AP = {ap:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Multi-class)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Save Model and Label Map
joblib.dump(clf, 'xgb_cls_embeddings.pkl')
joblib.dump({0: "Negative", 1: "Neutral", 2: "Positive"}, 'label_map.pkl')
