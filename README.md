# Stock Sentiment Prediction - Week 1

This project analyzes news headlines and stock price data for major tech companies to understand how news sentiment, topics, and events impact market movements.  
It combines Natural Language Processing (NLP), topic modeling, event detection, and quantitative financial analysis.

---

## Project Structure

- `src/`: Source code modules  
  - `Data_Loader.py`: Functions for loading and preprocessing news data  
  - `Data_Cleaner.py`: Data cleaning and quality check utilities  
  - `topic_modeling.py`: Topic modeling and event extraction functionality  
  - `Quantitative_Analysis.py`: Technical indicators and financial metrics  
- `scripts/`:  
  - `sentiment_analysis.py`: Sentiment extraction, merging with stock data, and comprehensive analysis/visualization  
- `notebooks/`: Analysis notebooks  
  - `Task1_EDA.ipynb`: Exploratory Data Analysis  
  - `Task1_Topic_Modeling.ipynb`: Topic and event analysis  
  - `Task2_Quantitative_Analysis.ipynb`: Technical and quantitative stock analysis  
  - `Task3_Sentiment_Analysis.ipynb`: Sentiment, event, and price movement analysis  
- `tests/`: Unit tests and test data  
- `Data/`:  
  - `data-week1/`: Raw stock and news data  
  - `cleaned/`: Cleaned and processed data  
  - `analyzed/`: Analysis results and outputs  

---

## Setup

1. **Install required packages:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn nltk textblob pytest
    ```
    *For technical indicators, you may also need `TA-Lib` or similar libraries.*

2. **Data structure:**
    - Place raw data in `Data/data-week1/`
    - Cleaned data will be saved in `Data/cleaned/`
    - Analysis results in `Data/analyzed/`

---

## Analysis Workflow

1. **Exploratory Data Analysis (EDA)**
    - Descriptive statistics of news headlines
    - Time series and publisher analysis

2. **Topic Modeling & Event Extraction**
    - LDA topic modeling on headlines
    - Extraction and frequency analysis of key events (e.g., earnings, FDA approvals)

3. **Quantitative Stock Analysis**
    - Calculation of technical indicators (MA, RSI, MACD, etc.)
    - Risk and return metrics

4. **Sentiment Analysis & Correlation**
    - Sentiment scoring of headlines
    - Merging with stock data and technical indicators
    - Visualization of sentiment, returns, events, and topics

---

## How to Run

- Use the Jupyter notebooks in `/notebooks` for step-by-step analysis and visualization.
- Source code modules in `/src` and `/scripts` provide reusable functions and classes.
- Run tests with:
    ```bash
    pytest
    ```

---

## Example Visualizations

- Headline length and publisher distribution
- Frequency of specific news events
- Top keywords per topic (aggregate bar plot)
- Comprehensive company-wise plots: price, sentiment, events, and topic distribution

---

## Testing

See [`tests/README.md`](tests/README.md) for details on running and extending the test suite.

---

## Authors

- [Adoniyas TIbebe]

---
