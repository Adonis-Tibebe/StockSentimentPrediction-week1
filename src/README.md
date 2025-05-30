# Source Code Modules

## Data_Loader.py
- `load_data()`: Loads news data with optional company filtering
- Handles date parsing and basic data structure

## Data_Cleaner.py
- `check_data_quality()`: Performs quality checks on loaded data
- `clean_data()`: Removes duplicates, handles missing values
- `save_cleaned_data()`: Saves processed data to CSV

## topic_modeling.py
- Text vectorization and LDA topic modeling
- Event extraction and topic assignment
- Functions for analyzing topic evolution

### Usage Example
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