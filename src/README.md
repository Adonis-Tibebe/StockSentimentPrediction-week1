# Source Code Modules

This directory contains the main source code for data loading, cleaning, topic modeling, and quantitative analysis.

---

## Modules

### Data_Loader.py
- `load_data()`: Loads news data with optional company filtering.
- Handles date parsing and basic data structure.

### Data_Cleaner.py
- `check_data_quality()`: Performs quality checks on loaded data.
- `clean_data()`: Removes duplicates, handles missing values.
- `save_cleaned_data()`: Saves processed data to CSV.

### topic_modeling.py
- Text vectorization and LDA topic modeling.
- Event extraction and topic assignment.
- Functions for analyzing topic evolution.

### Quantitative_Analysis.py
- Loads and processes stock price data.
- Calculates technical indicators (MA, RSI, MACD, etc.).
- Computes advanced financial metrics and risk measures.
- Visualization functions for technical analysis.

---

## Usage Example

```python
from Data_Loader import load_data
from Data_Cleaner import clean_data
from topic_modeling import make_vectorizer, run_lda

# Load and clean data
data = load_data("path/to/data.csv")
cleaned_data = clean_data(data)

# Run topic modeling
vectorizer = make_vectorizer()
topics = run_lda(cleaned_data)
```

---
