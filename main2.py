import streamlit as st
from textblob import TextBlob
from text2emotion import get_emotion
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources for sentiment analysis
nltk.download('vader_lexicon')

# Function to analyze sentiment using NLTK SentimentIntensityAnalyzer
def get_sentiment_nltk(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Set page title with custom CSS
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            color: #3366cc;
            text-align: center;
            margin-bottom: 20px;
        }

        .emotion-section {
            background-color: #f7f7f7;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .emotion-header {
            font-size: 24px;
            color: #333333;
            margin-bottom: 10px;
        }

        .emotion-label {
            font-size: 18px;
            color: #555555;
            margin-bottom: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.markdown('<p class="title">Text Analysis and Correction</p>', unsafe_allow_html=True)

# Text input
user_text = st.text_area("Enter your text:")

if user_text:
    # Perform sentiment analysis with TextBlob
    blob = TextBlob(user_text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # Perform sentiment analysis with NLTK
    nltk_sentiment = get_sentiment_nltk(user_text)

    # Perform emotion detection
    emotions = get_emotion(user_text)

    # Display sentiment analysis results
    st.subheader("Sentiment Analysis:")
    st.write(f"Polarity (TextBlob): {sentiment_polarity:.2f}")
    st.write(f"Subjectivity (TextBlob): {sentiment_subjectivity:.2f}")

    # Display sentiment with an emoji for positive sentiment
    sentiment_emoji = "😊" if nltk_sentiment == "Positive" else "😐" if nltk_sentiment == "Neutral" else "😢"
    st.write(f"Sentiment (NLTK): {nltk_sentiment} {sentiment_emoji}")

    # Display emotion detection results with custom CSS
    st.markdown('<div class="emotion-section">', unsafe_allow_html=True)
    st.subheader("Emotion Detection:")
    for emotion, score in emotions.items():
        st.markdown(f'<p class="emotion-label">{emotion.capitalize()}: {score:.2f}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Spelling correction
    corrected_text = blob.correct()

    # Display corrected text
    st.subheader("Corrected Text:")
    st.write(corrected_text)

# About section
st.sidebar.subheader("About")
st.sidebar.write(
    "This is a simple Streamlit app for text analysis and correction. "
    "It performs sentiment analysis using TextBlob and NLTK, provides spelling correction, "
    "and detects emotions using text2emotion."
)
