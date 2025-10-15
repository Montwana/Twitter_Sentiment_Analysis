import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns


'''# Load cleaned dataset
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
print("Text vectorization complete")

# Save vectorizer
joblib.dump(vectorizer, "twitter_vectorizer_logistic.pkl")
print("TF-IDF vectorizer saved as 'twitter_vectorizer_logistic.pkl'")

# Train Logistic Regression
model = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)
model.fit(X_train, y_train)
print("Logistic Regression training complete")

# Evaluate on validation set
y_val_pred = model.predict(X_val)
print("\nValidation Performance:")
print(classification_report(y_val, y_val_pred, digits=4))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Evaluate on test set
#y_test_pred = model.predict(X_test)
#print("\nTest Performance:")
#print(classification_report(y_test, y_test_pred, digits=4))
#print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Save the trained model
#joblib.dump(model, "twitter_logistic_model.pkl")
#print("Logistic Regression model saved as 'twitter_logistic_model.pkl'")
'''



# Load and Split Data
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

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)
print("Text vectorization complete")

#os.makedirs("visuals", exist_ok=True)
#joblib.dump(vectorizer, "twitter_vectorizer_logistic.pkl")
#print("TF-IDF vectorizer saved as 'twitter_vectorizer_logistic.pkl'")

# Define Optuna Objective
def objective(trial):
    C = trial.suggest_loguniform("C", 0.01, 100)
    penalty = trial.suggest_categorical("penalty", ["l2"])  # 'l1' possible with solver='liblinear'
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
    max_iter = trial.suggest_categorical("max_iter", [1000, 5000, 10000])
    
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1).mean()
    return score

# Run Hyperparameter Tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best hyperparameters found:", study.best_params)
print("Best CV F1 macro score:", study.best_value)

# Train Final Logistic Regression Model
best_model = LogisticRegression(
    C=study.best_params["C"],
    penalty=study.best_params["penalty"],
    solver=study.best_params["solver"],
    class_weight="balanced",
    max_iter=study.best_params["max_iter"],
    random_state=42
)
best_model.fit(X_train, y_train)
print("Final Logistic Regression model trained with best hyperparameters")


#Visuals
# Define class names
classes = ['Negative', 'Neutral', 'Positive']

y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Validation Performance
print("Validation Performance:\n")
print(classification_report(y_val, y_val_pred, digits=4))  # Added line
precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(y_val, y_val_pred, average=None)
accuracy_val = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy_val)
print()

# Test Performance
print("Test Performance:\n")
print(classification_report(y_test, y_test_pred, digits=4))  # Added line
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average=None)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy_test)



# Validation vs Test Bar Graph
x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x - width, precision_val, width, label='Precision (Val)', color='skyblue')
plt.bar(x, recall_val, width, label='Recall (Val)', color='lightgreen')
plt.bar(x + width, f1_val, width, label='F1-score (Val)', color='salmon')

# Overlay Test bars with dashed outline
plt.bar(x - width, precision_test, width, fill=False, edgecolor='blue', linestyle='--', label='Precision (Test)')
plt.bar(x, recall_test, width, fill=False, edgecolor='green', linestyle='--', label='Recall (Test)')
plt.bar(x + width, f1_test, width, fill=False, edgecolor='red', linestyle='--', label='F1-score (Test)')

plt.xticks(x, classes)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title(f"Validation vs Test Metrics per Class\nVal Acc: {accuracy_val:.4f} | Test Acc: {accuracy_test:.4f}")
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig("visuals/logistic_val_vs_test_metrics.png")
plt.close()
print("Validation vs Test metrics diagram saved as 'visuals/logistic_val_vs_test_metrics.png'")

# Confusion Matrix for Test Set
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix — Logistic Regression (Test Set)')
plt.tight_layout()
#plt.savefig("visuals/logistic_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as 'visuals/logistic_confusion_matrix.png'")

# Word Influence / Top Features
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = best_model.coef_
top_n = 20  # top features per class

for i, class_label in enumerate(classes):
    coef = coefs[i]
    top_positive_indices = coef.argsort()[::-1][:top_n]
    top_negative_indices = coef.argsort()[:top_n]

    top_positive_words = feature_names[top_positive_indices]
    top_positive_values = coef[top_positive_indices]

    top_negative_words = feature_names[top_negative_indices]
    top_negative_values = coef[top_negative_indices]

    # Combine positive and negative words
    words = np.concatenate([top_positive_words[::-1], top_negative_words[::-1]])
    values = np.concatenate([top_positive_values[::-1], top_negative_values[::-1]])
    colors = ['green']*top_n + ['red']*top_n  # green=positive, red=negative

    plt.figure(figsize=(10,6))
    plt.barh(words, values, color=colors)
    plt.title(f"Top Influential Words - {class_label} Class")
    plt.xlabel("Coefficient Value (Influence)")
    plt.tight_layout()
    plt.savefig(f"visuals/logistic_word_influence_{class_label}.png")
    plt.close()
    print(f"Word influence diagram saved for class '{class_label}'")

#joblib.dump(best_model, "twitter_logistic_optuna.pkl")
#print("Tuned Logistic Regression model saved as 'twitter_logistic_optuna.pkl'")
