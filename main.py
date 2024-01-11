import pandas as pd
import streamlit as st
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import swifter  # Ensure swifter is installed: pip install swifter

# Set a custom directory for NLTK data download
nltk.data.path.append('path/to/custom/nltk_data')
nltk.download('vader_lexicon')

st.header('Sentiment Analysis')

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(str(text))
    return {'polarity': round(blob.sentiment.polarity, 2), 'subjectivity': round(blob.sentiment.subjectivity, 2)}

# Function to analyze sentiment using SentimentIntensityAnalyzer
def analyze_sentiment_nltk(review):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(review)
    return scores

# File upload section
upl = st.file_uploader('Upload McDonald\'s Reviews CSV file', type=['csv'])

if upl:
    # Read the CSV file with a specified encoding
    df = pd.read_csv(upl, encoding='latin1')

    # Display the loaded data
    st.write("Loaded Data:")
    st.write(df.head())

    # Analyze sentiment with TextBlob
    st.write("Sentiment Analysis (TextBlob):")
    df['textblob_sentiment'] = df['review'].apply(analyze_sentiment_textblob)
    st.write(df[['review', 'textblob_sentiment']])

    # Analyze sentiment with NLTK SentimentIntensityAnalyzer
    st.write("Sentiment Analysis (NLTK):")
    df['nltk_sentiment'] = df['review'].swifter.apply(analyze_sentiment_nltk)

    # Extract compound scores
    df['nltk_compound_sentiment'] = df['nltk_sentiment'].apply(lambda x: x['compound'] if pd.notnull(x) else None)

    st.write(df[['review', 'nltk_compound_sentiment']])

    # Plot sentiment distribution for TextBlob
    st.write("Sentiment Distribution (TextBlob):")
    st.bar_chart(df['textblob_sentiment'].apply(lambda x: x['polarity']).value_counts())

    # Plot sentiment distribution for NLTK
    st.write("Sentiment Distribution (NLTK):")
    st.bar_chart(df['nltk_compound_sentiment'].value_counts())

    # Download the analyzed data
    @st.cache_data
    def convert_df(df):
        # Reset the index to avoid duplicate labels
        df_reset = df.reset_index(drop=True)
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df_reset.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download Analyzed Data as CSV",
        data=csv,
        file_name='analyzed_reviews.csv',
        mime='text/csv',
    )
