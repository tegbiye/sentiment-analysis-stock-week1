import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
# nltk.download('vader_lexicon')


def get_sentiment_word(text):
    """Determine sentiment of the text as Positive, Negative, or Neutral."""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']

    return score


def get_sentiment(text):
    """Calculate sentiment polarity of the text."""
    return TextBlob(text).sentiment.polarity


def articles_sentiment_analysis(articles):
    """Perform sentiment analysis on articles and plot the results."""

    sentiment_counts = articles['sentiment_score_word'].value_counts(
    ).sort_index()

    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                palette=colors, alpha=0.8)
    plt.title('Sentiment Analysis of Articles')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)

    plt.show()


def get_sentiment_analysis_publisher(data, target_publisher):
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    # Filter data for the target publisher
    publisher_data = data[data['publisher'] == target_publisher]
    sentiment_counts = publisher_data['sentiment_score_word'].value_counts(
    ).sort_index()

    sentiment_counts.plot(kind="bar", figsize=(10, 4), title=f'Sentiment Analysis of {target_publisher}',
                          xlabel='Sentiment categories', ylabel='Number of Published Articles',
                          color=[colors[category] for category in sentiment_counts.index])
