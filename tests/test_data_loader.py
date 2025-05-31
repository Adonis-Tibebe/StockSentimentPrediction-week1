import pytest
import pandas as pd
from src.Data_Loader import load_data

@pytest.fixture
def sample_data_path():
    return "test_data/sample_news.csv"

def test_load_data_basic():
    """Test basic data loading functionality"""
    df = load_data("test_data/sample_news.csv")
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['headline', 'date', 'stock'])

def test_load_data_with_companies():
    """Test loading data with company filtering"""
    companies = ["AAPL", "GOOG"]
    df = load_data("test_data/sample_news.csv", companies)
    assert all(df['stock'].isin(companies))

def test_load_data_date_parsing():
    """Test date parsing in loaded data"""
    df = load_data("test_data/sample_news.csv")
    assert pd.api.types.is_datetime64_any_dtype(df['date'])

def test_load_data_invalid_path():
    """Test handling of invalid file path"""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv") 