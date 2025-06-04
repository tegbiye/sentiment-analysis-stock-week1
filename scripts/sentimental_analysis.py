import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import talib as tl
import plotly.express as pe
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
    # Filter data for the target publisher
    publisher_data = data[data['publisher'] == target_publisher]
    sentiment_counts = publisher_data['sentiment_score_word'].value_counts(
    ).sort_index()

    print(sentiment_counts)

    sentiment_counts.plot(kind="bar", figsize=(10, 4), title=f'Sentiment Analysis of {target_publisher}',
                          xlabel='Sentiment categories', ylabel='Number of Published Articles',
                          color=[colors[category] for category in sentiment_counts.index])


def load_historical_data(ticker):
    stock_data = pd.read_csv(
        f'../data/yfinance_data/{ticker}_historical_data.csv')
    return stock_data


def calculate_Descriptive_Statistics_Historical_Data(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tsla):
    aapl_stats = stock_data_aapl['Close'].describe().to_frame('AAPL')
    goog_stats = stock_data_goog['Close'].describe().to_frame('GOOG')
    amzn_stats = stock_data_amzn['Close'].describe().to_frame('AMZN')
    msft_stats = stock_data_msft['Close'].describe().to_frame('MSFT')
    meta_stats = stock_data_meta['Close'].describe().to_frame('META')
    nvda_stats = stock_data_nvda['Close'].describe().to_frame('NVDA')
    tsla_stats = stock_data_tsla['Close'].describe().to_frame('TSLA')
    combined_stats = pd.concat([aapl_stats, goog_stats, amzn_stats,
                               msft_stats, meta_stats, nvda_stats, tsla_stats], axis=1)
    return combined_stats


def combine_dataframes_checking_missing_values(*dfs):
    clean_dfs = [df for df in dfs if not df.isnull().values.any()]
    combined_df = pd.concat(clean_dfs, axis=0)
    return combined_df


def analysisClosingPriceWithDate(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tesla):
    # Create subplots for side-by-side display
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Adjust figsize as needed

    axs[0, 0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['Close'], label='Close', color='green')
    axs[0, 0].set_title('AAPL')
    axs[0, 0].legend()

    axs[0, 1].plot(stock_data_amzn['Date'],
                   stock_data_amzn['Close'], label='AMZN')
    axs[0, 1].set_title('AMZN')
    axs[0, 1].legend()

    axs[0, 2].plot(stock_data_goog['Date'],
                   stock_data_goog['Close'], label='Close', color='yellow')
    axs[0, 2].set_title('GOOG')
    axs[0, 2].legend()

    axs[1, 0].plot(stock_data_nvda['Date'],
                   stock_data_nvda['Close'], label='Close', color='brown')
    axs[1, 0].set_title('NVDA')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Date')

    axs[1, 1].plot(stock_data_msft['Date'],
                   stock_data_msft['Close'], label='Close', color='purple')
    axs[1, 1].set_title('MSFT')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Date')

    axs[1, 2].plot(stock_data_meta['Date'],
                   stock_data_meta['Close'], label='Close', color='orange')
    axs[1, 2].set_title('META')
    axs[1, 2].legend()
    axs[1, 2].set_xlabel('Date')

    axs[1, 2].plot(stock_data_tesla['Date'],
                   stock_data_tesla['Close'], label='Close', color='orange')
    axs[1, 2].set_title('TESLA')
    axs[1, 2].legend()
    axs[1, 2].set_xlabel('Date')

    plt.show()


def calculateTechnicalIndicator(stock_data):
    stock_data['SMA'] = tl.SMA(stock_data['Close'], timeperiod=20)
    stock_data['RSI'] = tl.RSI(stock_data['Close'], timeperiod=14)
    stock_data['EMA'] = tl.EMA(stock_data['Close'], timeperiod=20)

    macd_signal, macd, _ = tl.MACD(stock_data['Close'])
    stock_data['MACD'] = macd
    stock_data['MACD_Signal'] = macd_signal


def showTechnicalIndicator(stock_data, ticker, indicator):
    if (f'{stock_data}'.find(ticker)):
        fig = pe.line(stock_data[indicator], stock_data['Date'],
                      y=indicator, title=f'{indicator} {ticker}')
        fig.show()


def technicalIndicatorsVsClosingPrice(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tsla, ticker):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Adjust figsize as needed

    axs[0, 0].plot(stock_data_aapl['Date'], stock_data_aapl['Close'],
                   label='Closing price', color='green')
    axs[0, 0].plot(stock_data_aapl['Date'],
                   stock_data_aapl[ticker], label=ticker, color='red')
    axs[0, 0].set_title('AAPL')
    axs[0, 0].legend()

    axs[0, 1].plot(stock_data_amzn['Date'],
                   stock_data_amzn['Close'], label='Closing price')
    axs[0, 1].plot(stock_data_amzn['Date'],
                   stock_data_amzn[ticker], label=ticker, color='red')
    axs[0, 1].set_title('AMZN')
    axs[0, 1].legend()

    axs[0, 2].plot(stock_data_goog['Date'], stock_data_goog['Close'],
                   label='Closing price', color='yellow')
    axs[0, 2].plot(stock_data_goog['Date'],
                   stock_data_goog[ticker], label=ticker, color='red')
    axs[0, 2].set_title('GOOG')
    axs[0, 2].legend()

    axs[1, 0].plot(stock_data_nvda['Date'], stock_data_nvda['Close'],
                   label='Closing price', color='blue')
    axs[1, 0].plot(stock_data_nvda['Date'],
                   stock_data_nvda[ticker], label=ticker, color='red')
    axs[1, 0].set_title('NVDA')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Date')

    axs[1, 1].plot(stock_data_msft['Date'], stock_data_msft['Close'],
                   label='Closing price', color='purple')
    axs[1, 1].plot(stock_data_msft['Date'],
                   stock_data_msft[ticker], label=ticker, color='red')
    axs[1, 1].set_title('MSFT')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Date')

    axs[1, 2].plot(stock_data_meta['Date'], stock_data_meta['Close'],
                   label='Closing price', color='pink')
    axs[1, 2].plot(stock_data_meta['Date'],
                   stock_data_meta[ticker], label=ticker, color='red')
    axs[1, 2].set_title('META')
    axs[1, 2].legend()
    axs[1, 2].set_xlabel('Date')

    axs[1, 2].plot(stock_data_tsla['Date'], stock_data_tsla['Close'],
                   label='Closing price', color='pink')
    axs[1, 2].plot(stock_data_tsla['Date'],
                   stock_data_tsla[ticker], label=ticker, color='red')
    axs[1, 2].set_title('TSLA')
    axs[1, 2].legend()
    axs[1, 2].set_xlabel('Date')

    plt.show()


def closingPriceRelativeStrengthIndex(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tsla):
    fig, axs = plt.subplots(6, 2, gridspec_kw={"height_ratios": [
                            1, 1, 1, 1, 1, 1]}, figsize=(16, 22))

    # For AAPL
    axs[0][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['Close'], label="Close")
    axs[0][0].set_title("AAPL Stock Price")
    axs[0][0].legend()
    axs[1][0].axhline(y=70, color='r', linestyle="--")
    axs[1][0].axhline(y=30, color='g', linestyle="--")
    axs[1][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['RSI'], color='orange', label="RSI")
    axs[1][0].legend()

    # for GOOG
    axs[0][1].plot(stock_data_goog['Date'],
                   stock_data_goog['Close'], label="Close")
    axs[0][1].set_title("GOOG Stock Price")
    axs[0][1].legend()
    axs[1][1].axhline(y=70, color='r', linestyle="--")
    axs[1][1].axhline(y=30, color='g', linestyle="--")
    axs[1][1].plot(stock_data_goog['Date'],
                   stock_data_goog['RSI'], color='orange', label="RSI")
    axs[1][1].legend()

    # for AMZN
    axs[2][0].plot(stock_data_amzn['Date'],
                   stock_data_amzn['Close'], label="Close")
    axs[2][0].set_title("AMZN Stock Price")
    axs[2][0].legend()
    axs[3][0].axhline(y=70, color='r', linestyle="--")
    axs[3][0].axhline(y=30, color='g', linestyle="--")
    axs[3][0].plot(stock_data_amzn['Date'],
                   stock_data_amzn['RSI'], color='orange', label="RSI")
    axs[3][0].legend()

    # for NVDA
    axs[2][1].plot(stock_data_nvda['Date'],
                   stock_data_nvda['Close'], label="Close")
    axs[2][1].set_title("NVDA Stock Price")
    axs[2][1].legend()
    axs[3][1].axhline(y=70, color='r', linestyle="--")
    axs[3][1].axhline(y=30, color='g', linestyle="--")
    axs[3][1].plot(stock_data_nvda['Date'],
                   stock_data_nvda['RSI'], color='orange', label="RSI")
    axs[3][1].legend()

    # for MSFT
    axs[4][0].plot(stock_data_msft['Date'],
                   stock_data_msft['Close'], label="Close")
    axs[4][0].set_title("MSFT Stock Price")
    axs[4][0].legend()
    axs[5][0].axhline(y=70, color='r', linestyle="--")
    axs[5][0].axhline(y=30, color='g', linestyle="--")
    axs[5][0].plot(stock_data_msft['Date'],
                   stock_data_msft['RSI'], color='orange', label="RSI")
    axs[5][0].legend()

    # for META
    axs[4][1].plot(stock_data_meta['Date'],
                   stock_data_meta['Close'], label="Close")
    axs[4][1].set_title("META Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=70, color='r', linestyle="--")
    axs[5][1].axhline(y=30, color='g', linestyle="--")
    axs[5][1].plot(stock_data_meta['Date'],
                   stock_data_meta['RSI'], color='orange', label="RSI")
    axs[5][1].legend()

    # for TESLA
    axs[4][1].plot(stock_data_tsla['Date'],
                   stock_data_tsla['Close'], label="Close")
    axs[4][1].set_title("TESLA Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=70, color='r', linestyle="--")
    axs[5][1].axhline(y=30, color='g', linestyle="--")
    axs[5][1].plot(stock_data_tsla['Date'],
                   stock_data_tsla['RSI'], color='orange', label="RSI")
    axs[5][1].legend()
    fig.show()
    # momentum oscillator that measures the speed and change of price movements.
    # Identifying overbought and oversold conditions. A reading above 70 is typically
    # considered overbought, while a reading below 30 is considered oversold.


def calculatePortfolioWeightAndPerformance():
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    tickers = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'NVDA', 'META', 'TSLA']
    # Load data from each ticker file
    dataframes = [load_historical_data(ticker) for ticker in tickers]

    # Combine dataframes into a single DataFrame
    combined_data = pd.concat(dataframes, axis=1)['Close']

    new_column_names = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
    combined_data.columns = new_column_names

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(combined_data)
    S = risk_models.sample_cov(combined_data)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    weights = dict(zip(['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA'], [
                   round(value, 2) for value in weights.values()]))

    # Print Portfolio weights
    print("Portfolio Weights:")
    print(weights)

    # Calculate and print portfolio performance
    print("\nPortfolio Performance:")
    ef.portfolio_performance(verbose=True)


def closingPriceMovingAverageConvergenceDivergence(stock_data_aapl, stock_data_amzn, stock_data_goog, stock_data_meta, stock_data_msft, stock_data_nvda, stock_data_tsla):

    fig, axs = plt.subplots(6, 2, gridspec_kw={"height_ratios": [
                            1, 1, 1, 1, 1, 1]}, figsize=(16, 22))

    # for AAPL
    axs[0][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['Close'], label="Close")
    axs[0][0].set_title("AAPL Stock Price")
    axs[0][0].legend()
    axs[1][0].axhline(y=5, color='r', linestyle="--")
    axs[1][0].axhline(y=-5, color='g', linestyle="--")
    axs[1][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[1][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[1][0].legend()

    # for GOOG
    axs[0][1].plot(stock_data_goog['Date'],
                   stock_data_goog['Close'], label="Close")
    axs[0][1].set_title("GOOG Stock Price")
    axs[0][1].legend()
    axs[1][1].axhline(y=5, color='r', linestyle="--")
    axs[1][1].axhline(y=-5, color='g', linestyle="--")
    axs[1][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[1][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[1][1].legend()

    # for AMZN
    axs[2][0].plot(stock_data_amzn['Date'],
                   stock_data_amzn['Close'], label="Close")
    axs[2][0].set_title("AMZN Stock Price")
    axs[2][0].legend()
    axs[3][0].axhline(y=5, color='r', linestyle="--")
    axs[3][0].axhline(y=-5, color='g', linestyle="--")
    axs[3][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[3][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[3][0].legend()

    # for NVDA
    axs[2][1].plot(stock_data_nvda['Date'],
                   stock_data_nvda['Close'], label="Close")
    axs[2][1].set_title("NVDA Stock Price")
    axs[2][1].legend()
    axs[3][1].axhline(y=5, color='r', linestyle="--")
    axs[3][1].axhline(y=-5, color='g', linestyle="--")
    axs[3][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[3][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[3][1].legend()

    # for MSFT
    axs[4][0].plot(stock_data_msft['Date'],
                   stock_data_msft['Close'], label="Close")
    axs[4][0].set_title("MSFT Stock Price")
    axs[4][0].legend()
    axs[5][0].axhline(y=5, color='r', linestyle="--")
    axs[5][0].axhline(y=-5, color='g', linestyle="--")
    axs[5][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[5][0].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[5][0].legend()

    # for META
    axs[4][1].plot(stock_data_meta['Date'],
                   stock_data_meta['Close'], label="Close")
    axs[4][1].set_title("META Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=5, color='r', linestyle="--")
    axs[5][1].axhline(y=-5, color='g', linestyle="--")
    axs[5][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[5][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[5][1].legend()

    # for META
    axs[4][1].plot(stock_data_tsla['Date'],
                   stock_data_tsla['Close'], label="Close")
    axs[4][1].set_title("TESLA Stock Price")
    axs[4][1].legend()
    axs[5][1].axhline(y=5, color='r', linestyle="--")
    axs[5][1].axhline(y=-5, color='g', linestyle="--")
    axs[5][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD'], color='orange', label="MACD")
    axs[5][1].plot(stock_data_aapl['Date'],
                   stock_data_aapl['MACD_Signal'], color='r', label="MACD_Signal")
    axs[5][1].legend()
    fig.show()
