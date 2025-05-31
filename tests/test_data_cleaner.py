import pytest
import pandas as pd
import numpy as np
from src.Data_Cleaner import check_data_quality, clean_data, save_cleaned_data

@pytest.fixture
def sample_dirty_data():
    return pd.DataFrame({
        'headline': ['Test 1', 'Test 2', 'Test 1', None],
        'date': ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-03'],
        'stock': ['AAPL', 'GOOG', 'AAPL', 'MSFT']
    })

def test_check_data_quality(sample_dirty_data):
    """Test data quality checking"""
    quality_report = check_data_quality(sample_dirty_data)
    assert isinstance(quality_report, dict)
    assert 'duplicates' in quality_report
    assert 'missing_values' in quality_report

def test_clean_data(sample_dirty_data):
    """Test data cleaning functionality"""
    cleaned = clean_data(sample_dirty_data)
    assert len(cleaned) < len(sample_dirty_data)  # Should remove duplicates
    assert cleaned['headline'].isna().sum() == 0  # Should handle missing values

def test_save_cleaned_data(tmp_path, sample_dirty_data):
    """Test saving cleaned data"""
    file_path = tmp_path / "test_cleaned.csv"
    save_cleaned_data(sample_dirty_data, str(file_path))
    assert file_path.exists()
    loaded = pd.read_csv(file_path)
    assert len(loaded) == len(sample_dirty_data) 