# Stock Sentiment Prediction - Week 1

This project analyzes news headlines data to understand patterns, topics, and potential market impacts.

## Project Structure
- `src/`: Source code modules
  - `Data_Loader.py`: Functions for loading and preprocessing data
  - `Data_Cleaner.py`: Data cleaning and quality check utilities
  - `topic_modeling.py`: Topic modeling and event extraction functionality

- `notebooks/`: Analysis notebooks
  - `Task1_EDA.ipynb`: Exploratory Data Analysis
  - `Task1_Topic_Modeling.ipynb`: Topic and event analysis

- `tests/`: Test files
  - Unit tests for source code modules
  - Test data and utilities

## Setup
1. Install required packages:
```bash
pip install pandas numpy scikit-learn pytest
```

2. Data structure:
- Place raw data in `Data/data-week1/`
- Cleaned data will be saved in `Data/cleaned/`
- Analysis results in `Data/analyzed/`

## Analysis Components
1. Exploratory Data Analysis (EDA)
   - Descriptive statistics
   - Time series analysis
   - Publisher analysis

2. Topic Modeling
   - Event extraction
   - Topic discovery
   - Temporal analysis
