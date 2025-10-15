import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter

# Load dataset
df = pd.read_csv("balanced_sentiment_dataset.csv", encoding='utf-8', on_bad_lines='skip')
print("Columns:", df.columns)

# Download NLTK resources (only the first time)
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Initialize tools
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}
lemmatizer = WordNetLemmatizer()

# Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()

    # Remove URLs, mentions, hashtags, HTML entities, and non-breaking spaces
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+|&\w+;|<.*?>', '', text)

    # Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Remove numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Normalize repeated letters (e.g., "agaaaaainnnn" → "again")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Tokenize, remove stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return ' '.join(tokens)

# Apply Cleaning
tqdm.pandas(desc="Cleaning texts")
df['text'] = df['text'].progress_apply(clean_text)  # replace existing column

# Drop rows where text is empty or NaN
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']

# Encode Sentiments
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

# Keep only 'text' and 'label' columns
df = df[['text', 'label']]

# Save cleaned dataset
df.to_csv("cleaned_sentiment_dataset.csv", index=False)
print(" Cleaned dataset saved as 'cleaned_sentiment_dataset.csv' with columns: text, label")

# Visualization
os.makedirs("visuals", exist_ok=True)

# Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Label (0=Negative,1=Neutral,2=Positive)")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("visuals/sentiment_distribution.png")
plt.close()

# Word Cloud per sentiment
sentiment_mapping = {0:"Negative", 1:"Neutral", 2:"Positive"}
for label, name in sentiment_mapping.items():
    text_data = " ".join(df[df['label']==label]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {name} Tweets")
    plt.tight_layout()
    plt.savefig(f"visuals/wordcloud_{name.lower()}.png")
    plt.close()

# Tweet length distribution
df['tweet_length'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,5))
sns.histplot(df['tweet_length'], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visuals/tweet_length_distribution.png")
plt.close()

print("All visualizations saved in the 'visuals' folder")
