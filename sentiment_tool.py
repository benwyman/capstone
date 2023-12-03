import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import pipeline

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Ensure the spaCy model is downloaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Global variables for stopwords and keywords
stop = stopwords.words('english')
keywords = ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'stock', 'market',
            'invest', 'trade', 'equity', 'share', 'dividend']
stop += keywords


def analyze_data(df):
    print("Analyzing...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by='date', inplace=True)

    df = df.drop_duplicates(subset=['title'])

    if 'paragraph_text' in df.columns:
        df.rename(columns={'paragraph_text': 'text'}, inplace=True)
    elif 'snippet' in df.columns:
        df['text'] = df['snippet']
    else:
        print("Error: The DataFrame does not have the required text columns.")
        return

    df['text'] = df['text'].astype(str).apply(lambda x: " ".join(x.lower().split()))
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    df['lemmatized_text'] = df['text'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    df['sentiment_transformers'] = df['lemmatized_text'].apply(lambda x: sentiment_pipeline(x[:512])[0]['score'])

    df['polarity'] = df['lemmatized_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['score'] = np.where(df['polarity'] > 0, df['sentiment_transformers'], -df['sentiment_transformers'])

    def label_sentiment(row):
        if row['score'] > 0:
            return 'Positive'
        elif row['score'] < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df.apply(label_sentiment, axis=1)
    return df


def graph_sentiment_by_date(original_df):
    df = original_df.copy()  # Create a copy of the DataFrame
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    average_sentiment_per_week = df['score'].resample('W').mean().reset_index()
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=average_sentiment_per_week, x='date', y='score', marker='o')
    plt.title('Average Sentiment Score per Week')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    return fig


def graph_sentiment_frequency(original_df):
    df = original_df.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df['score'], fill=True, ax=ax)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Sentiment Scores')
    return fig


def graph_polarity_distribution(original_df):
    df = original_df.copy()
    fig = plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df['polarity'], fill=True, color="skyblue")
    plt.xlabel('Polarity')
    plt.ylabel('Density')
    plt.title('Distribution of Polarity')
    plt.xlim(-1, 1)
    return fig


def graph_polarity_date(original_df):
    df = original_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    average_polarity_per_week = df['polarity'].resample('W').mean().reset_index()
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=average_polarity_per_week, x='date', y='polarity', marker='o')
    plt.title('Average Polarity per Week')
    plt.xlabel('Date')
    plt.ylabel('Average Polarity')
    plt.xticks(rotation=45)
    return fig


def graph_volume(original_df):
    df = original_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    texts_per_week = df.resample('W').size().reset_index(name='count')
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=texts_per_week, x='date', y='count')
    plt.title('Volume of Texts Over Time (Weekly)')
    plt.xlabel('Date')
    plt.ylabel('Number of Texts')
    plt.xticks(rotation=45)
    return fig


def summarize_sentiment_data(df):
    if 'sentiment' not in df.columns:
        print("Error: The DataFrame does not have a 'sentiment' column.")
        return

    summary = {'total_texts': len(df)}

    # Counts and ratios for each sentiment type
    sentiment_counts = df['sentiment'].value_counts()
    summary['sentiment_counts'] = sentiment_counts.to_dict()

    # Percent ratio of positive/negative sentiment
    positive_ratio = (sentiment_counts.get('Positive', 0) / summary['total_texts']) * 100
    negative_ratio = (sentiment_counts.get('Negative', 0) / summary['total_texts']) * 100
    summary['positive_ratio'] = positive_ratio
    summary['negative_ratio'] = negative_ratio

    # Average sentiment score
    summary['average_sentiment_score'] = df['score'].mean()

    # Print the summary
    print("Sentiment Analysis Summary:")
    print(f"Total Texts: {summary['total_texts']}")
    print(f"Sentiment Counts: {summary['sentiment_counts']}")
    print(f"Positive Sentiment Ratio: {summary['positive_ratio']:.2f}%")
    print(f"Negative Sentiment Ratio: {summary['negative_ratio']:.2f}%")
    print(f"Average Sentiment Score: {summary['average_sentiment_score']:.2f}")

    return summary
