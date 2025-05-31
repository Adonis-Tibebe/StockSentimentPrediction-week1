import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from typing import Dict, Tuple, Any

def headline_length_stats(df: pd.DataFrame, headline_col: str = 'headline') -> Tuple[Dict[str, pd.Series], Figure]:
    """
    Calculate descriptive statistics for headline lengths in both characters and words.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data
        headline_col (str): Name of the headline column (default: 'headline')
        
    Returns:
        Tuple[Dict[str, pd.Series], Figure]: 
            - Dictionary containing descriptive statistics for both character and word counts
            - Figure object containing the subplots for both distributions
            
    Raises:
        ValueError: If headline column is not found in DataFrame
    """
    if headline_col not in df.columns:
        raise ValueError(f"Column '{headline_col}' not found in DataFrame")
    
    # Calculate lengths
    char_lengths = df[headline_col].str.len()
    word_lengths = df[headline_col].str.split().str.len()
    
    # Calculate descriptive statistics
    stats = {
        'characters': char_lengths.describe(),
        'words': word_lengths.describe()
    }
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Character length histogram
    sns.histplot(data=pd.DataFrame({'length': char_lengths}), x='length', bins=50, ax=ax1)
    ax1.set_title('Distribution of Headline Character Lengths')
    ax1.set_xlabel('Number of Characters')
    ax1.set_ylabel('Count')
    
    # Add mean and median lines for characters
    char_mean = float(stats['characters']['mean'])
    char_median = float(stats['characters']['50%'])
    ax1.axvline(char_mean, color='red', linestyle='--', 
                label=f'Mean: {char_mean:.1f}')
    ax1.axvline(char_median, color='green', linestyle='--', 
                label=f'Median: {char_median:.1f}')
    ax1.legend()
    
    # Word count histogram
    sns.histplot(data=pd.DataFrame({'length': word_lengths}), x='length', bins=30, ax=ax2)
    ax2.set_title('Distribution of Headline Word Counts')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Count')
    
    # Add mean and median lines for words
    word_mean = float(stats['words']['mean'])
    word_median = float(stats['words']['50%'])
    ax2.axvline(word_mean, color='red', linestyle='--', 
                label=f'Mean: {word_mean:.1f}')
    ax2.axvline(word_median, color='green', linestyle='--', 
                label=f'Median: {word_median:.1f}')
    ax2.legend()
    
    plt.tight_layout()
    return stats, fig

def count_by_publisher(df: pd.DataFrame, publisher_col: str = 'publisher', top_n: int = 20) -> Tuple[pd.Series, Figure]:
    """
    Count articles per publisher and create a bar chart.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data
        publisher_col (str): Name of the publisher column (default: 'publisher')
        top_n (int): Number of top publishers to show (default: 20)
        
    Returns:
        Tuple[pd.Series, Figure]:
            - Value counts of articles per publisher
            - Figure object containing the bar chart
            
    Raises:
        ValueError: If publisher column is not found in DataFrame
    """
    if publisher_col not in df.columns:
        raise ValueError(f"Column '{publisher_col}' not found in DataFrame")
    
    # Get value counts
    publisher_counts = df[publisher_col].value_counts()
    
    # Select top N publishers
    top_publishers = publisher_counts.head(top_n)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_publishers.values, y=top_publishers.index, ax=ax)
    ax.set_title(f'Top {top_n} Publishers by Number of Articles')
    ax.set_xlabel('Number of Articles')
    ax.set_ylabel('Publisher')
    
    plt.tight_layout()
    return publisher_counts, fig

def time_series_counts(
    df: pd.DataFrame,
    date_col: str = 'date',
    freq: str = 'D',
    rolling_window: int = 7
) -> Tuple[pd.Series, Figure]:
    """
    Analyze publication date trends and create a time series plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data
        date_col (str): Name of the date column (default: 'date')
        freq (str): Frequency for grouping ('D' for daily, 'W' for weekly, etc.)
        rolling_window (int): Window size for rolling average (default: 7)
        
    Returns:
        Tuple[pd.Series, Figure]:
            - Time series of article counts
            - Figure object containing the line plot
            
    Raises:
        ValueError: If date column is not found or not datetime type
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Column '{date_col}' must be datetime type")
    
    # Group by date and count
    daily_counts = df.groupby(pd.Grouper(key=date_col, freq=freq)).size()  # type: ignore
    
    # Calculate rolling average
    rolling_avg = daily_counts.rolling(window=rolling_window, center=True).mean()
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(15, 6))
    daily_counts.plot(alpha=0.5, label='Daily Counts', ax=ax)
    rolling_avg.plot(color='red', label=f'{rolling_window}-day Rolling Average', ax=ax)
    
    ax.set_title('Number of Articles Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Articles')
    ax.legend()
    
    plt.tight_layout()
    return daily_counts, fig  # type: ignore

def analyze_news_data(df: pd.DataFrame, output_dir: str = "../outputs") -> Dict[str, Any]:
    """
    Run all analyses and save plots to files.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data
        output_dir (str): Directory to save output plots (default: "../outputs")
        
    Returns:
        Dict[str, Any]: Dictionary containing all analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Headline length analysis
    length_stats, length_fig = headline_length_stats(df)
    length_fig.savefig(os.path.join(output_dir, 'headline_lengths.png'))
    results['headline_stats'] = length_stats
    
    # Publisher analysis
    publisher_counts, pub_fig = count_by_publisher(df)
    pub_fig.savefig(os.path.join(output_dir, 'publisher_counts.png'))
    results['publisher_counts'] = publisher_counts
    
    # Time series analysis
    time_series, time_fig = time_series_counts(df)
    time_fig.savefig(os.path.join(output_dir, 'time_series.png'))
    results['time_series'] = time_series
    
    plt.close('all')  # Close all figures to free memory
    
    return results 