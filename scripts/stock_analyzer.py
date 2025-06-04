import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")


class StockAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None

    @property
    def info(self) -> Dict:
        if self._info is None:
            self._info = self.stock.info
        return self._info

    def get_historical_data(self, period: str = "1y", interval: str = "1d",
                            start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical stock data.
        :param period: Period for which to fetch data (e.g., '1y', '6mo', '1mo').
        :param interval: Data interval (e.g., '1d', '1wk', '1mo').
        :param start: Start date in 'YYYY-MM-DD' format.
        :param end: End date in 'YYYY-MM-DD' format.
        :return: DataFrame with historical stock data.
        """
        if start and end:
            data = self.stock.history(start=start, end=end, interval=interval)
        else:
            # Default to period if no start/end provided
            data = self.stock.history(period=period, interval=interval)
        return data

    def get_news_impact_analysis(self, news_date: str, days_before: int = 5, days_after: int = 5) -> pd.DataFrame:
        """
        Analyze the impact of news on stock price.
        :param news_date: Date of the news in 'YYYY-MM-DD' format.
        :param days_before: Number of days before the news to consider.
        :param days_after: Number of days after the news to consider.
        :return: DataFrame with stock prices around the news date.
        """
        start_date = (datetime.strptime(news_date, '%Y-%m-%d') -
                      timedelta(days=days_before)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(news_date, '%Y-%m-%d') +
                    timedelta(days=days_after)).strftime('%Y-%m-%d')
        return self.get_historical_data(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the stock data.
        :param data: DataFrame with stock data.
        :return: DataFrame with added technical indicators.
        """
        df = data.copy()

        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def plot_stock_data(self, data: pd.DataFrame, indicators: bool = True, volume: bool = True) -> None:
        """
        Plot stock data with optional technical indicators.
        :param data: DataFrame with stock data.
        :param indicators: List of indicators to plot (e.g., ['SMA_20', 'SMA_50', 'RSI']).
        """
        fig = make_subplots(rows=2 if volume else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.7, 0.3] if volume else [1])

        # Plot stock price chart
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='OHLC'), row=1, col=1)
        # Plot indicators if provided
        if indicators:
            # Add moving averages
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(
                color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(
                color='red')), row=1, col=1)
        if volume:
            # Add volume bar chart
            fig.add_trace(go.Bar(
                x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        # Add layout details
        fig.update_layout(title=f"{self.ticker} Stock Price",
                          xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        fig.show()

    def analyze_news_impact(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """
        Analyze the impact of news on stock prices.
        :param news_data: DataFrame with news data.
        :param price_data: DataFrame with stock price data.

        :return: Dictionary with analysis results.
        """
        analysis_results = {}
        for _, news in news_data.iterrows():
            news_date = pd.to_datetime(news['date'])
            # Get price data around news event
            event_data = self.get_news_impact_analysis(
                news_date.strftime('%Y-%m-%d'))

            if not event_data.empty:
                # Calculate price changes
                price_before = event_data['Close'].iloc[0]
                price_after = event_data['Close'].iloc[-1]
                price_change = (
                    (price_after - price_before) / price_before) * 100

                # Calculate the volatility
                volatility = event_data['Close'].pct_change().std() * 100

                analysis_results[news_date.strftime('%Y-%m-%d')] = {
                    "headline": news['headline'],
                    "price_change": price_change,
                    "volatility": volatility,
                    "volume_change": (
                        (event_data['Volume'].iloc[-1] - event_data['Volume'].iloc[0]
                         ) / event_data['Volume'].iloc[0] * 100
                    )}

        return analysis_results
