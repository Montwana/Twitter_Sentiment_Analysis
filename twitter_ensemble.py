import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load cleaned dataset
df = pd.read_csv("cleaned_sentiment_dataset.csv")  

# Split dataset into train, validation, and test
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train size: {len(X_train_text)}")
print(f"Validation size: {len(X_val_text)}")
print(f"Test size: {len(X_test_text)}")

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)
print("Text vectorization complete")

# Save vectorizer
#os.makedirs("models", exist_ok=True)
#joblib.dump(vectorizer, "models/twitter_vectorizer_ensemble.pkl")
#print("TF-IDF vectorizer saved as 'models/twitter_vectorizer_ensemble.pkl'")

# Initialize classifiers
svm_model = LinearSVC(C=0.235, max_iter=10000, class_weight='balanced', random_state=42)
logistic_model = LogisticRegression(C=1.0, max_iter=10000, class_weight='balanced', random_state=42)
decision_tree_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=42)

# Ensemble using VotingClassifier (soft voting for probabilities)
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('logistic', logistic_model),
        ('tree', decision_tree_model)
    ],
    voting='hard'  
)


# Train ensemble
ensemble.fit(X_train, y_train)
print("Ensemble training complete")

# Evaluate on validation and test sets
def evaluate_model(model, X, y, set_name="Set"):
    y_pred = model.predict(X)
    print(f"\n{set_name} Performance:")
    print(classification_report(y, y_pred, digits=4))
    accuracy = accuracy_score(y, y_pred)
    print(f"{set_name} Accuracy: {accuracy:.4f}")
    return y_pred, accuracy

y_val_pred, val_acc = evaluate_model(ensemble, X_val, y_val, "Validation")
y_test_pred, test_acc = evaluate_model(ensemble, X_test, y_test, "Test")



# Confusion matrix
classes = ['Negative', 'Neutral', 'Positive']
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Test Set (Ensemble)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
#plt.savefig("visuals/ensemble_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as 'visuals/ensemble_confusion_matrix.png'")

# Save ensemble model
#joblib.dump(ensemble, "models/twitter_ensemble_model.pkl")
#print("Ensemble model saved as 'models/twitter_ensemble_model.pkl'")

# Accuracy/Precision/Recall/F1 bar chart
precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(y_val, y_val_pred, average=None)
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average=None)

x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10,6))
# Validation bars
plt.bar(x - width, precision_val, width, label='Precision (Val)', color='skyblue')
plt.bar(x, recall_val, width, label='Recall (Val)', color='lightgreen')
plt.bar(x + width, f1_val, width, label='F1-score (Val)', color='salmon')

# Test bars overlay (dashed outline)
plt.bar(x - width, precision_test, width, fill=False, edgecolor='blue', linestyle='--', label='Precision (Test)')
plt.bar(x, recall_test, width, fill=False, edgecolor='green', linestyle='--', label='Recall (Test)')
plt.bar(x + width, f1_test, width, fill=False, edgecolor='red', linestyle='--', label='F1-score (Test)')

plt.xticks(x, classes)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title(f"Validation vs Test Metrics per Class\nVal Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig("visuals/ensemble_val_vs_test_metrics.png")
plt.close()
print("Validation vs Test metrics diagram saved as 'visuals/ensemble_val_vs_test_metrics.png'")