import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
import optuna
import numpy as np


'''
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

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)
print(" Text vectorization complete")

# Train Linear SVM with balanced class weights
model = LinearSVC(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print(" SVM training complete")

# Evaluate on validation set
y_val_pred = model.predict(X_val)
print(" Validation Set Performance:")
print(classification_report(y_val, y_val_pred, digits=2))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Save vectorizer
#joblib.dump(vectorizer, "twitter_vectorizer.pkl")
#print(" TF-IDF vectorizer saved as 'twitter_vectorizer.pkl'")

# Evaluate on test set 
#y_test_pred = model.predict(X_test)
#print("\n Test Set Performance:")
#print(classification_report(y_test, y_test_pred, digits=4))
#print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Save the trained model
#joblib.dump(model, "twitter_svm_model.pkl")
#print(" Trained SVM model saved as 'twitter_svm_model.pkl'")
'''

#After tuning

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

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)
print(" Text vectorization complete")

# Save vectorizer
#joblib.dump(vectorizer, "twitter_vectorizer.pkl")
#print("TF-IDF vectorizer saved as 'twitter_vectorizer.pkl'")

# Bayesian Optimization using Optuna
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_loguniform('C', 0.01, 100)
    max_iter = trial.suggest_categorical('max_iter', [1000, 5000, 10000])
    
    # Initialize Linear SVC
    model = LinearSVC(C=C, max_iter=max_iter, class_weight='balanced', random_state=42)
    
    # 3-fold cross-validation on training set
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1).mean()
    return score

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

print("Best hyperparameters found:", study.best_params)
print("Best CV F1 macro score:", study.best_value)

# Train final model with best hyperparameters
best_model = LinearSVC(
    C=study.best_params['C'],
    max_iter=study.best_params['max_iter'],
    class_weight='balanced',
    random_state=42
)
best_model.fit(X_train, y_train)
print(" Final SVM trained with best hyperparameters")

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
print("\n Validation Performance:")
print(classification_report(y_val, y_val_pred, digits=4))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = best_model.predict(X_test)
print("\n Test Performance:")
print(classification_report(y_test, y_test_pred, digits=4))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Save the tuned model
#joblib.dump(best_model, "twitter_svm_model_tuned.pkl")
#print("Tuned SVM model saved as 'twitter_svm_model_tuned.pkl'")

#confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
classes = ['Negative', 'Neutral', 'Positive']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
#plt.savefig("visuals/confusion_matrix_Optuna_test.png")
plt.close()
print("Confusion matrix saved as 'visuals/confusion_matrix_Optuna_test.png'")


# Word Influence / Top Features

# Get feature names from TF-IDF vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Get SVM coefficients (shape: n_classes x n_features)
coefs = best_model.coef_

classes = ['Negative', 'Neutral', 'Positive']
top_n = 20

for i, class_label in enumerate(classes):
    coef = coefs[i]
    top_positive_indices = coef.argsort()[::-1][:top_n]
    top_negative_indices = coef.argsort()[:top_n]

    top_positive_words = feature_names[top_positive_indices]
    top_positive_values = coef[top_positive_indices]

    top_negative_words = feature_names[top_negative_indices]
    top_negative_values = coef[top_negative_indices]

    # Combine positive and negative words for plotting
    words = np.concatenate([top_positive_words[::-1], top_negative_words[::-1]])
    values = np.concatenate([top_positive_values[::-1], top_negative_values[::-1]])

    colors = ['green']*top_n + ['red']*top_n  # green for positive, red for negative

    plt.figure(figsize=(10,6))
    plt.barh(words, values, color=colors)
    plt.title(f"Top Influential Words - {class_label} Class")
    plt.xlabel("Coefficient Value (Influence)")
    plt.tight_layout()
    plt.savefig(f"visuals/word_influence_{class_label}.png")
    plt.close()
    print(f"Word influence diagram saved for class '{class_label}' as 'visuals/word_influence_{class_label}.png'")


#Accuracy validation vs test
# Metrics for Validation
y_val_pred = best_model.predict(X_val)
precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(y_val, y_val_pred, average=None)
accuracy_val = accuracy_score(y_val, y_val_pred)


# Metrics for Test
y_test_pred = best_model.predict(X_test)
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average=None)
accuracy_test = accuracy_score(y_test, y_test_pred)

# Plot side-by-side comparison
x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10,6))

# Validation bars
plt.bar(x - width, precision_val, width, label='Precision (Val)', color='skyblue')
plt.bar(x, recall_val, width, label='Recall (Val)', color='lightgreen')
plt.bar(x + width, f1_val, width, label='F1-score (Val)', color='salmon')

# Overlay Test bars (dashed edge)
plt.bar(x - width, precision_test, width, fill=False, edgecolor='blue', linestyle='--', label='Precision (Test)')
plt.bar(x, recall_test, width, fill=False, edgecolor='green', linestyle='--', label='Recall (Test)')
plt.bar(x + width, f1_test, width, fill=False, edgecolor='red', linestyle='--', label='F1-score (Test)')

plt.xticks(x, classes)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title(f"Validation vs Test Metrics per Class\nVal Acc: {accuracy_val:.4f} | Test Acc: {accuracy_test:.4f}")
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig("visuals/val_vs_test_metrics.png")
plt.close()

print("Validation vs Test metrics diagram saved as 'visuals/val_vs_test_metrics.png'")
