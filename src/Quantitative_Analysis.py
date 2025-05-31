import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Union
from pandas import DataFrame

class QuantitativeAnalyzer:
    """
    A class to handle quantitative analysis of stock data.
    """
    
    def __init__(self):
        """Initialize the QuantitativeAnalyzer with display settings."""
        # Configure display settings
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        
        # Set basic plot style instead of seaborn
        plt.style.use('default')
        
        # Set random seed for reproducibility
        np.random.seed(42)

    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[DataFrame]:
        """
        Load stock data using yfinance with error handling.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            Optional[DataFrame]: DataFrame containing the stock data or None if error occurs
        """
        try:
            # Validate inputs
            if not isinstance(symbol, str) or not isinstance(start_date, str) or not isinstance(end_date, str):
                raise ValueError("Symbol and dates must be strings")
            
            # Download stock data
            stock_data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            # Check if data was retrieved
            if not isinstance(stock_data, DataFrame) or stock_data.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in stock_data.columns for col in required_columns):
                print(f"Missing required columns for {symbol}")
                return None
                
            # Calculate daily returns
            stock_data['Daily_Return'] = stock_data['Close'].pct_change()
            
            return stock_data
            
        except Exception as e:
            print(f"Error loading stock data: {str(e)}")
            return None

    def plot_stock_analysis(self, stock_data: Optional[DataFrame], symbol: str) -> None:
        """
        Create basic stock analysis plots.
        
        Args:
            stock_data (Optional[DataFrame]): Stock price data
            symbol (str): Stock symbol for plot titles
        """
        # Validate input
        if stock_data is None or not isinstance(stock_data, DataFrame) or stock_data.empty:
            print("No valid data to plot")
            return
            
        try:
            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Stock price history
            ax1.plot(stock_data.index, stock_data['Close'], label='Close Price')
            ax1.set_title(f'{symbol} Stock Price History')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Daily returns distribution
            returns_data = stock_data['Daily_Return'].dropna()
            returns_df = pd.DataFrame(returns_data)  # Convert to DataFrame for seaborn
            sns.histplot(data=returns_df, x='Daily_Return', kde=True, ax=ax2)
            ax2.set_title('Distribution of Daily Returns')
            ax2.set_xlabel('Daily Returns')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")

    def calculate_metrics(self, stock_data: Optional[DataFrame]) -> Dict[str, float]:
        """
        Calculate various financial metrics from the stock data.
        
        Args:
            stock_data (Optional[DataFrame]): Stock price data
            
        Returns:
            Dict[str, float]: Dictionary containing calculated metrics
        """
        # Validate input
        if stock_data is None or not isinstance(stock_data, DataFrame) or stock_data.empty:
            print("No valid data for metric calculation")
            return {}
            
        try:
            # Get clean daily returns (remove NaN values)
            daily_returns = stock_data['Daily_Return'].dropna()
            
            if len(daily_returns) < 2:  # Need at least 2 points for calculations
                print("Insufficient data for metric calculation")
                return {}
            
            # Calculate metrics with error handling
            metrics: Dict[str, float] = {
                'daily_returns_mean': float(daily_returns.mean()),
                'daily_returns_std': float(daily_returns.std()),
                'annualized_volatility': float(daily_returns.std() * np.sqrt(252)),
                'total_return': float((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1),
                'sharpe_ratio': float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {} 