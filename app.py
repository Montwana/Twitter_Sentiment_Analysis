import streamlit as st
import joblib
import pandas as pd
import tweepy
from streamlit_lottie import st_lottie
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Load Model and Vectorizer
@st.cache_resource
def load_model_vectorizer():
    model = joblib.load("models/twitter_ensemble_model.pkl")
    vectorizer = joblib.load("models/twitter_vectorizer_ensemble.pkl")
    return model, vectorizer

model, vectorizer = load_model_vectorizer()

# Twitter API Setup
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    st.error("Missing BEARER_TOKEN. Set it in Streamlit secrets.")

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# Streamlit Config & Header Animation
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_twitter = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_iwmd6pyr.json")
lottie_confetti = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_neutral = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tll0j4bb.json")
lottie_rain = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_u4yrau.json")

st_lottie(lottie_twitter, height=150)
st.title("Twitter Sentiment Analyzer")
st.markdown("""
Enter a tweet manually or fetch a user's latest tweets for real-time sentiment analysis.
""")

# Helper Functions
def predict_sentiment(text, threshold=0.7):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    tokens = vectorizer.build_analyzer()(clean_text)
    known_tokens = [t for t in tokens if t in vectorizer.vocabulary_]

    if not known_tokens:
        return "Uncertain"

    X_input = vectorizer.transform([clean_text])
    try:
        probs = model.predict_proba(X_input)[0]
        max_prob = probs.max()
        predicted_class = probs.argmax()
    except AttributeError:
        predicted_class = model.predict(X_input)[0]
        max_prob = 1.0

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if max_prob < threshold:
        return "Uncertain"

    return sentiment_map.get(predicted_class, str(predicted_class))

def sentiment_color(sentiment):
    if sentiment == "Negative":
        return "#FF4C4C"
    elif sentiment == "Neutral":
        return "#4C6EFF"
    elif sentiment == "Positive":
        return "#28C76F"
    elif sentiment == "Uncertain":
        return "#FFA500"
    else:
        return "#FFFFFF"

def display_animated_card(text, sentiment):
    color = sentiment_color(sentiment)
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color} 0%, #000000 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.3);
            color: white;
            transition: transform 0.2s;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <h4>Sentiment: <b>{sentiment}</b></h4>
            <p>{text}</p>
        </div>
    """, unsafe_allow_html=True)

def display_lottie_for_sentiment(sentiment):
    if sentiment == "Positive":
        st_lottie(lottie_confetti, height=100)
    elif sentiment == "Neutral":
        st_lottie(lottie_neutral, height=100)
    elif sentiment == "Negative":
        st_lottie(lottie_rain, height=100)

def generate_wordcloud(texts):
    clean_texts = [re.sub(r'[^a-zA-Z\s]', '', t).strip() for t in texts if t.strip() != '']
    combined_text = ' '.join(clean_texts)

    if not combined_text.strip():
        st.info("No meaningful words found to generate a word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def plot_sentiment_summary_plotly(sentiments, title="Sentiment Summary", typed=False):
    filtered_sentiments = [s for s in sentiments if s != "Uncertain"]
    counts = pd.Series(filtered_sentiments).value_counts()
    sentiments_order = ["Positive", "Neutral", "Negative"]
    counts = counts.reindex(sentiments_order, fill_value=0)

    colors = {"Positive": "#28C76F", "Neutral": "#4C6EFF", "Negative": "#FF4C4C"}

    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=[colors[s] for s in counts.index],
        )
    )

    y_title = "Sentiment Strength" if typed else "Number of Tweets"

    fig.update_layout(
        title=title,
        xaxis_title="Sentiment",
        yaxis_title=y_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

# Cache for fetched tweets
@st.cache_data(show_spinner=False)
def fetch_user_tweets(username, num_tweets):
    try:
        user_resp = client.get_user(username=username)
        if user_resp.data is None:
            return None, "User not found."
        user_id = user_resp.data.id

        tweets_resp = client.get_users_tweets(id=user_id, max_results=num_tweets, tweet_fields=['text'])
        tweets_data = tweets_resp.data or []

        tweet_texts = [tweet.text for tweet in tweets_data]
        return tweet_texts, None
    except tweepy.errors.TooManyRequests:
        return None, "Rate limit reached. Please try again later."
    except Exception as e:
        return None, str(e)

# User Input Option
option = st.radio("Choose input method:", ["Type a Tweet", "Fetch User Tweets"])

# Type a Tweet
st.session_state.setdefault("user_input", "")
if option == "Type a Tweet":
    st.text_area("Enter Tweet Text:", height=120, key="user_input")

    if st.button("Analyze Sentiment"):
        if st.session_state.user_input.strip() == "":
            st.warning("Please enter a tweet to analyze.")
        else:
            sentiment = predict_sentiment(st.session_state.user_input)
            display_animated_card(st.session_state.user_input, sentiment)
            display_lottie_for_sentiment(sentiment)

            st.session_state["typed_tweets"] = [st.session_state.user_input]
            st.session_state["typed_sentiments"] = [sentiment]

            st.subheader("Tweets Summary")
            plot_sentiment_summary_plotly(st.session_state["typed_sentiments"], "Tweets Sentiment Summary", typed=True)

            st.subheader("Word Cloud of Tweets")
            generate_wordcloud(st.session_state["typed_tweets"])

# Fetch User Tweets
if option == "Fetch User Tweets":
    username = st.text_input("Enter Twitter Username (without @):")
    num_tweets = st.slider("Number of Tweets to Fetch:", 1, 50, 5)

    if st.button("Fetch & Analyze"):
        if username.strip() == "":
            st.warning("Please enter a Twitter username.")
        else:
            with st.spinner(f"Fetching {num_tweets} tweets from @{username}..."):
                tweet_texts, error = fetch_user_tweets(username, num_tweets)

                if error:
                    st.error(f"Error: {error}")
                elif not tweet_texts:
                    st.info("No tweets found for this user.")
                else:
                    st.success(f"Analyzing {len(tweet_texts)} tweets from @{username}:")
                    sentiments = [predict_sentiment(text) for text in tweet_texts]

                    for text, sentiment in zip(tweet_texts, sentiments):
                        display_animated_card(text, sentiment)
                        display_lottie_for_sentiment(sentiment)

                    st.session_state["fetched_tweets"] = tweet_texts
                    st.session_state["fetched_sentiments"] = sentiments

                    st.subheader(f"Fetched Tweets Sentiment Summary (@{username})")
                    plot_sentiment_summary_plotly(
                        st.session_state["fetched_sentiments"],
                        f"Fetched Tweets Sentiment Summary (@{username})",
                        typed=False
                    )

                    st.subheader(f"Word Cloud of @{username}'s Tweets")
                    generate_wordcloud(st.session_state["fetched_tweets"])
