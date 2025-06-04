import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from typing import Optional, Dict, Any, Union, List, Tuple
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
        
        # Define technical indicator parameters
        self.ma_periods = [20, 50, 200]  # Moving average periods
        self.rsi_period = 14  # RSI period
        self.macd_params = (12, 26, 9)  # MACD parameters (fast, slow, signal)
        self.bb_params = (20, 2)  # Bollinger Bands parameters (period, std)

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

    def calculate_technical_indicators(self, stock_data: Optional[DataFrame]) -> Optional[DataFrame]:
        """
        Calculate various technical indicators using TA-Lib.
        
        Args:
            stock_data (Optional[DataFrame]): Stock price data
            
        Returns:
            Optional[DataFrame]: DataFrame with added technical indicators
        """
        if stock_data is None or not isinstance(stock_data, DataFrame) or stock_data.empty:
            print("No valid data for technical analysis")
            return None
            
        try:
            df = stock_data.copy()
            
            # Get the stock symbol from the columns
            symbol = df.columns.get_level_values(1)[0]
            
            # Extract price data and convert to numpy arrays
            close = df[('Close', symbol)].to_numpy(dtype=np.float64)
            high = df[('High', symbol)].to_numpy(dtype=np.float64)
            low = df[('Low', symbol)].to_numpy(dtype=np.float64)
            
            # Calculate Moving Averages
            for period in self.ma_periods:
                ma = talib.SMA(close, timeperiod=period)
                df[(f'MA_{period}', '')] = ma
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            df[('RSI', '')] = rsi
            
            # Calculate MACD
            fast, slow, signal = self.macd_params
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            df[('MACD', '')] = macd
            df[('MACD_Signal', '')] = macd_signal
            df[('MACD_Hist', '')] = macd_hist
            
            # Calculate Bollinger Bands
            period, std = self.bb_params
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=period,
                nbdevup=std,
                nbdevdn=std
            )
            df[('BB_Upper', '')] = upper
            df[('BB_Middle', '')] = middle
            df[('BB_Lower', '')] = lower
            
            # Calculate ATR (Average True Range)
            atr = talib.ATR(
                high,
                low,
                close,
                timeperiod=14
            )
            df[('ATR', '')] = atr
            
            # Calculate Stochastic Oscillator
            k, d = talib.STOCH(
                high,
                low,
                close,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            df[('STOCH_K', '')] = k
            df[('STOCH_D', '')] = d
            
            # Note: The first few rows will contain NaN values because indicators
            # need some historical data to start calculating. This is normal.
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return None

    def plot_technical_analysis(self, stock_data: Optional[DataFrame], symbol: str) -> None:
        """
        Create comprehensive technical analysis plots.
        
        Args:
            stock_data (Optional[DataFrame]): Stock price data with technical indicators
            symbol (str): Stock symbol for plot titles
        """
        if stock_data is None or not isinstance(stock_data, DataFrame) or stock_data.empty:
            print("No valid data to plot")
            return
            
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 15))
            gs = fig.add_gridspec(3, 2)
            
            # 1. Price and Moving Averages
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(stock_data.index, stock_data['Close'], label='Close Price')
            for period in self.ma_periods:
                ax1.plot(stock_data.index, stock_data[f'MA_{period}'], 
                        label=f'{period}-day MA')
            ax1.set_title(f'{symbol} Price and Moving Averages')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # 2. RSI
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(stock_data.index, stock_data['RSI'])
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.axhline(y=30, color='g', linestyle='--')
            ax2.set_title('RSI (14)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('RSI')
            ax2.grid(True)
            
            # 3. MACD
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(stock_data.index, stock_data['MACD'], label='MACD')
            ax3.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal')
            ax3.bar(stock_data.index, stock_data['MACD_Hist'], label='Histogram')
            ax3.set_title('MACD')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True)
            
            # 4. Bollinger Bands
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.plot(stock_data.index, stock_data['Close'], label='Close')
            ax4.plot(stock_data.index, stock_data['BB_Upper'], label='Upper BB')
            ax4.plot(stock_data.index, stock_data['BB_Middle'], label='Middle BB')
            ax4.plot(stock_data.index, stock_data['BB_Lower'], label='Lower BB')
            ax4.set_title('Bollinger Bands')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Price')
            ax4.legend()
            ax4.grid(True)
            
            # 5. Stochastic Oscillator
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(stock_data.index, stock_data['STOCH_K'], label='%K')
            ax5.plot(stock_data.index, stock_data['STOCH_D'], label='%D')
            ax5.axhline(y=80, color='r', linestyle='--')
            ax5.axhline(y=20, color='g', linestyle='--')
            ax5.set_title('Stochastic Oscillator')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Value')
            ax5.legend()
            ax5.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating technical analysis plots: {str(e)}")

    def calculate_advanced_metrics(self, stock_data: Optional[DataFrame]) -> Dict[str, float]:
        """
        Calculate advanced financial metrics and risk measures.
        
        Args:
            stock_data (Optional[DataFrame]): Stock price data
            
        Returns:
            Dict[str, float]: Dictionary containing calculated metrics
        """
        if stock_data is None or not isinstance(stock_data, DataFrame) or stock_data.empty:
            print("No valid data for metric calculation")
            return {}
            
        try:
            # Get clean daily returns (remove NaN values)
            daily_returns = stock_data['Daily_Return'].dropna()
            
            if len(daily_returns) < 2:
                print("Insufficient data for metric calculation")
                return {}
            
            # Calculate risk-adjusted returns
            risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
            excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate metrics
            metrics: Dict[str, float] = {
                # Basic metrics
                'daily_returns_mean': float(daily_returns.mean()),
                'daily_returns_std': float(daily_returns.std()),
                'annualized_volatility': float(daily_returns.std() * np.sqrt(252)),
                'total_return': float((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1),
                
                # Risk-adjusted metrics
                'sharpe_ratio': float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)),
                'sortino_ratio': float((daily_returns.mean() / daily_returns[daily_returns < 0].std()) * np.sqrt(252)),
                
                # Risk metrics
                'max_drawdown': float(self._calculate_max_drawdown(stock_data['Close'])),
                'var_95': float(np.percentile(daily_returns, 5)),  # 95% VaR
                'cvar_95': float(daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()),  # 95% CVaR
                
                # Technical metrics
                'rsi_latest': float(stock_data['RSI'].iloc[-1]),
                'macd_latest': float(stock_data['MACD'].iloc[-1]),
                'bb_width': float((stock_data['BB_Upper'].iloc[-1] - stock_data['BB_Lower'].iloc[-1]) / 
                                stock_data['BB_Middle'].iloc[-1])
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate the maximum drawdown from peak."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min()) 