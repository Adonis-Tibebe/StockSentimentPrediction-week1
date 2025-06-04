# Scripts Directory

This folder contains high-level scripts for advanced analysis, sentiment extraction, and data merging.

---

## Scripts

### sentiment_analysis.py
- **Purpose:**  
  Performs sentiment analysis on news headlines, merges sentiment data with stock price data, and generates comprehensive visualizations.
- **Key Features:**  
  - Sentiment scoring using NLP (TextBlob, NLTK)
  - Entity and event extraction from headlines
  - Aggregation and alignment of news and stock data
  - Calculation of correlations between sentiment, returns, and technical indicators
  - Multi-panel plotting for company-wise analysis

---

## Usage

These scripts are typically imported and used within the Jupyter notebooks in the `/notebooks` directory, but can also be run or extended independently for batch processing or automation.

---

## Example

```python
from scripts.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.load_news_data("path/to/news.csv")
analyzer.load_stock_data("path/to/stock_data/")
analyzer.analyze_daily_sentiment()
analyzer.plot_sentiment_vs_returns("AAPL")
```

---

## Notes

- Ensure all dependencies are installed (see main project README).
- Scripts are designed to be modular and reusable for future weeks and extended analyses.

---