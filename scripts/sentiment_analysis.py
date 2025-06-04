import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.blob import BaseBlob
from typing import Dict, List, Optional, Tuple, Literal
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
from src.Quantitative_Analysis import QuantitativeAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ast

class SentimentAnalyzer:
    """Class for analyzing sentiment correlation between news and stock movements."""
    
    def __init__(self):
        """Initialize the SentimentAnalyzer with company mappings and NLTK resources."""
        # Define company symbol mappings (original symbol -> news data symbol)
        self.company_mappings = {
            'AAPL': 'AAPL',
            'AMZN': 'AMZN',
            'GOOG': 'GOOG',
            'META': 'FB',    # Special case: META is FB in news data
            'MSFT': 'MSF',   # Special case: MSFT is MSF in news data
            'TSLA': 'TSLA',
            'NVDA': 'NVDA'
        }
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: Could not initialize NLTK components: {str(e)}")
            self.stop_words = set()
            self.lemmatizer = None
        
        # Initialize quantitative analyzer
        self.quant_analyzer = QuantitativeAnalyzer()
        
        # Initialize data storage
        self.news_data = None
        self.stock_data = {}
        self.aligned_data = {}
        self.technical_indicators = {}
        
    def load_news_data(self, filepath: str) -> None:
        """
        Load the pre-processed news data that includes topics and events.
        
        Args:
            filepath (str): Path to the news data CSV file
        """
        try:
            self.news_data = pd.read_csv(filepath)
            # Convert date column to datetime
            self.news_data['date'] = pd.to_datetime(self.news_data['date'])
            print(f"Successfully loaded news data with shape: {self.news_data.shape}")
        except Exception as e:
            print(f"Error loading news data: {str(e)}")
            
    def load_stock_data(self, base_path: str) -> None:
        """
        Load stock price data for all companies from local CSV files and calculate technical indicators.
        
        Args:
            base_path (str): Base path to the directory containing stock data files in format {symbol}_historical_data.csv
        """
        for symbol in self.company_mappings.keys():
            try:
                # Construct file path for the symbol
                file_path = os.path.join(base_path, f"{symbol}_historical_data.csv")
                
                # Load data from CSV
                stock_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                
                if stock_data is not None and not stock_data.empty:
                    # Create multi-index columns to match yfinance format
                    stock_data.columns = pd.MultiIndex.from_product([stock_data.columns, [symbol]])
                    
                    # Calculate daily returns
                    stock_data[('Daily_Return', '')] = stock_data[('Close', symbol)].pct_change()
                    
                    # Calculate technical indicators using QuantitativeAnalyzer
                    stock_data_with_indicators = self.quant_analyzer.calculate_technical_indicators(stock_data)
                    if stock_data_with_indicators is not None:
                        self.stock_data[symbol] = stock_data_with_indicators
                        # Calculate advanced metrics
                        metrics = self.quant_analyzer.calculate_advanced_metrics(stock_data_with_indicators)
                        print(f"Calculated metrics for {symbol}: {metrics}")
                    print(f"Successfully loaded {symbol} stock data with technical indicators from {file_path}")
                else:
                    print(f"No data found in file for {symbol}")
            except FileNotFoundError:
                print(f"Stock data file not found for {symbol} at {file_path}")
            except Exception as e:
                print(f"Error loading {symbol} stock data: {str(e)}")
                
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using NLTK for better sentiment analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Tokenize
            tokens = word_tokenize(str(text).lower())
            
            # Remove stopwords and lemmatize
            if self.lemmatizer is not None:
                tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in self.stop_words and token.isalnum()
                ]
            else:
                tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Text preprocessing failed: {str(e)}")
            return str(text)

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using both TextBlob and enhanced preprocessing.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Sentiment polarity score (-1 to 1)
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Calculate sentiment
            return float(TextBlob(processed_text).sentiment.polarity)  # type: ignore
        except:
            return 0.0
            
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using NLTK.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of extracted entities
        """
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(str(text))
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract proper nouns (potential entities)
            entities = [
                word for word, pos in pos_tags 
                if pos in ['NNP', 'NNPS']  # Proper nouns
            ]
            
            return entities
        except Exception as e:
            print(f"Warning: Entity extraction failed: {str(e)}")
            return []

    def analyze_daily_sentiment(self) -> None:
        """Calculate daily sentiment scores with enhanced text analysis and robust merging."""
        if self.news_data is None:
            print("News data not loaded")
            return

        for symbol, news_symbol in self.company_mappings.items():
            print(f"\nProcessing {symbol} (News symbol: {news_symbol})")

            # Filter news for the company
            company_news = self.news_data[self.news_data['stock'] == news_symbol].copy()
            company_news['date'] = pd.to_datetime(company_news['date'])
            print(company_news.dtypes)

            if company_news.empty:
                print(f"No news data found for {symbol}")
                continue

            # Print date range of news data
            news_start = company_news['date'].min()
            news_end = company_news['date'].max()
            print(f"News data range: {news_start.date()} to {news_end.date()}")

            # Calculate sentiment and extract entities
            company_news['sentiment'] = company_news['headline'].apply(self.calculate_sentiment)
            company_news['entities'] = company_news['headline'].apply(self.extract_entities)

            # Group by date and calculate aggregates
            daily_news = company_news.groupby('date').agg({
                'sentiment': ['mean', 'count'],
                'topics': lambda x: x.mode().iloc[0] if not x.empty else [],
                'specific_events': lambda x: list(set([item for sublist in x for item in sublist])),
                'entities': lambda x: list(set([item for sublist in x for item in sublist]))
            })

            # Flatten column names
            daily_news.columns = ['Sentiment', 'NewsCount', 'DailyTopics', 'DailyEvents', 'DailyEntities']

            # Ensure index is datetime and named 'Date'
            daily_news = daily_news.reset_index()
            daily_news['date'] = pd.to_datetime(daily_news['date'])
            daily_news = daily_news.set_index('date')
            daily_news.index.name = 'Date'

            # Merge with stock data
            if symbol in self.stock_data:
                stock_data = self.stock_data[symbol].copy()
                # Flatten MultiIndex columns if present
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = [
                        '_'.join([str(i) for i in col if i]) for col in stock_data.columns.values
                    ]
                # Rename columns for consistency
                for col in stock_data.columns:
                    if col.lower() == 'close' or col.lower().startswith('close_'):
                        stock_data = stock_data.rename(columns={col: 'Close'})
                    if col.lower() == 'daily_return' or col.lower().startswith('daily_return'):
                        stock_data = stock_data.rename(columns={col: 'Returns'})

                stock_data.index = pd.to_datetime(stock_data.index)
                stock_data.index.name = 'Date'

                print(stock_data.index.dtype)
                print("-----------------------------------------------------------------------------")
                # Print date range of stock data
                stock_start = stock_data.index.min()
                stock_end = stock_data.index.max()
                print(f"Stock data range: {stock_start.date()} to {stock_end.date()}")

                # Find overlapping date range
                start_date = max(news_start, stock_start)
                end_date = min(news_end, stock_end)
                print(f"Overlapping range: {start_date.date()} to {end_date.date()}")

                # Restrict both dataframes to the overlapping date range
                daily_news_overlap = daily_news.loc[start_date:end_date]
                stock_data_overlap = stock_data.loc[start_date:end_date]

                # Reindex stock data to match news dates exactly
                news_dates = daily_news_overlap.index
                reindexed_stock = stock_data_overlap.reindex(news_dates)

                # For numeric columns, interpolate missing values
                numeric_cols = reindexed_stock.select_dtypes(include=['float64', 'int64']).columns
                reindexed_stock[numeric_cols] = reindexed_stock[numeric_cols].interpolate(method='linear')

                print(f"Reindexed stock data shape: {reindexed_stock.shape}")
                print(f"News data shape: {daily_news_overlap.shape}")

                # Print sample of both datasets
                print("\nReindexed Stock Data Sample (first 3 rows):")
                print(reindexed_stock.head(3))
                print("\nNews Data Sample (first 3 rows):")
                print(daily_news_overlap.head(3))

                # Merge the aligned data
                merged_data = reindexed_stock.join(daily_news_overlap, how='left')

                # Forward fill and handle missing values
                merged_data['Sentiment'] = merged_data['Sentiment'].fillna(method='ffill')
                merged_data['NewsCount'] = merged_data['NewsCount'].fillna(0)
                merged_data['DailyTopics'] = merged_data['DailyTopics'].fillna(str([]))
                merged_data['DailyEvents'] = merged_data['DailyEvents'].fillna(str([]))
                merged_data['DailyEntities'] = merged_data['DailyEntities'].fillna(str([]))

                self.aligned_data[symbol] = merged_data
                print(f"\nSuccessfully processed {symbol} data - {len(merged_data)} days")    
                
    def calculate_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between sentiment, returns, and technical indicators.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of correlation metrics for each company
        """
        results = {}
        
        for symbol, data in self.aligned_data.items():
            # Basic correlations
            sentiment_return_corr, sentiment_return_p = stats.pearsonr(
                data['Sentiment'].fillna(0),
                data['Returns'].fillna(0)
            )
            
            # Correlations with technical indicators
            indicator_correlations = {}
            for col in ['RSI', 'MACD', 'BB_Width']:
                if col in data.columns:
                    corr, p_val = stats.pearsonr(
                        data['Sentiment'].fillna(0),
                        data[col].fillna(0)
                    )
                    indicator_correlations[f'{col}_correlation'] = corr
                    indicator_correlations[f'{col}_p_value'] = p_val
            
            results[symbol] = {
                'sentiment_return_correlation': sentiment_return_corr,
                'sentiment_return_p_value': sentiment_return_p,
                **indicator_correlations
            }
            
        return results
        
    def plot_sentiment_vs_returns(self, symbol: str) -> None:
        """
        Create comprehensive visualization including technical indicators.

        Args:
            symbol (str): Company symbol to plot
        """
        if symbol not in self.aligned_data:
            print(f"No data available for {symbol}")
            return

        # Ensure index is named 'Date' before resetting
        self.aligned_data[symbol].index.name = 'Date'
        data = self.aligned_data[symbol].reset_index()

        # Robustly find the date column
        if 'Date' in data.columns:
            date_col = 'Date'
        elif 'date' in data.columns:
            date_col = 'date'
        elif 'index' in data.columns:
            date_col = 'index'
        else:
            print("No date column found! Columns are:", data.columns)
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 2)

        # Plot 1: Price, MA, and Sentiment
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data[date_col], data['Close'], label='Close Price', alpha=0.7)
        if 'MA_20' in data.columns:
            ax1.plot(data[date_col], data['MA_20'], label='20-day MA', alpha=0.7)
        ax1_2 = ax1.twinx()
        ax1_2.plot(data[date_col], data['Sentiment'], label='Sentiment', color='red', alpha=0.5)
        ax1.set_title(f'{symbol} Price and Sentiment')
        ax1.legend(loc='upper left')
        ax1_2.legend(loc='upper right')

        # Plot 2: Returns vs Sentiment scatter
        ax2 = fig.add_subplot(gs[1, 0])
        if 'Returns' in data.columns:
            sns.regplot(data=data, x='Sentiment', y='Returns', ax=ax2)
            ax2.set_title('Returns vs Sentiment')
        else:
            ax2.text(0.5, 0.5, 'No Returns data', ha='center', va='center')
            ax2.set_title('Returns vs Sentiment')

        # Plot 3: RSI with Sentiment
        ax3 = fig.add_subplot(gs[1, 1])
        if 'RSI' in data.columns:
            ax3.plot(data[date_col], data['RSI'], label='RSI', alpha=0.7)
            ax3_2 = ax3.twinx()
            ax3_2.plot(data[date_col], data['Sentiment'], label='Sentiment', color='red', alpha=0.5)
            ax3.set_title('RSI vs Sentiment')
            ax3.legend(loc='upper left')
            ax3_2.legend(loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'No RSI data', ha='center', va='center')
            ax3.set_title('RSI vs Sentiment')

        # Plot 4: Event Analysis
        ax4 = fig.add_subplot(gs[2, :])
        event_data = data[data['DailyEvents'] != '[]']
        ax4.plot(data[date_col], data['Close'], label='Price', alpha=0.3)
        if not event_data.empty:
            ax4.scatter(event_data[date_col], event_data['Close'],
                        c=event_data['Sentiment'], cmap='RdYlGn',
                        label='Events', s=100)
        ax4.set_title('Price with Event Markers (color=sentiment)')

        # Plot 5: Topic Distribution
        ax5 = fig.add_subplot(gs[3, :])
        # Fix: Convert string representations of lists to actual lists
        data['DailyTopics'] = data['DailyTopics'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        topic_counts = data['DailyTopics'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        topic_counts.plot(kind='bar', ax=ax5)
        ax5.set_title('Distribution of Daily Topics')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()