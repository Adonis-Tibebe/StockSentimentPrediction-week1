import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
import os

def check_data_quality(df: pd.DataFrame) -> Dict[str, Union[int, Dict]]:
    """
    Check for data quality issues in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to check
        
    Returns:
        Dict[str, Union[int, Dict]]: Dictionary containing quality metrics:
            - total_rows: Total number of rows in the DataFrame
            - duplicates: Number of duplicate rows
            - missing_values: Dictionary with count of missing values per column
            - data_types: Dictionary with data type of each column
            - unique_values: Dictionary with count of unique values per column
            
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    try:
        quality_report = {
            'total_rows': len(df),
            'duplicates': df.duplicated().sum(),
            'missing_values': {
                col: df[col].isna().sum() 
                for col in df.columns
            },
            'data_types': {
                col: str(df[col].dtype)
                for col in df.columns
            },
            'unique_values': {
                col: df[col].nunique()
                for col in df.columns
            }
        }
        
        # Check for specific issues in date column if it exists
        if 'date' in df.columns:
            quality_report['date_range'] = {
                'min_date': df['date'].min(),
                'max_date': df['date'].max(),
                'null_dates': df['date'].isna().sum()
            }
            
        return quality_report
        
    except Exception as e:
        raise RuntimeError(f"Error checking data quality: {str(e)}")

def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    remove_na: bool = True,
    subset: Optional[list] = None,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Clean the DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows (default: True)
        remove_na (bool): Whether to remove rows with missing values (default: True)
        subset (Optional[list]): List of columns to consider for duplicate removal
                               and NA checking. If None, uses all columns.
        date_column (str): Name of the date column for specific date handling
                          (default: 'date')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty or if specified columns don't exist
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if subset is not None and not all(col in df.columns for col in subset):
        missing_cols = [col for col in subset if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    try:
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Remove duplicates if requested
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates(subset=subset)
            duplicates_removed = initial_rows - len(df_clean)
            print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values if requested
        if remove_na:
            # Get initial count
            pre_na_rows = len(df_clean)
            
            # Remove rows with missing values in specified columns
            if subset:
                df_clean = df_clean.dropna(subset=subset)
            else:
                df_clean = df_clean.dropna()
                
            na_removed = pre_na_rows - len(df_clean)
            print(f"Removed {na_removed} rows with missing values")
        
        # Special handling for date column if it exists
        if date_column in df_clean.columns:
            # Sort by date
            df_clean = df_clean.sort_values(date_column)
            
            # Report date range
            print(f"Date range: {df_clean[date_column].min()} to {df_clean[date_column].max()}")
        
        total_removed = initial_rows - len(df_clean)
        print(f"Total rows removed: {total_removed} ({(total_removed/initial_rows)*100:.2f}%)")
        
        return df_clean
        
    except Exception as e:
        raise RuntimeError(f"Error cleaning data: {str(e)}")

def save_cleaned_data(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = "../Data/cleaned",
    index: bool = False
) -> str:
    """
    Save the cleaned DataFrame to a CSV file in the specified directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the file (with or without .csv extension)
        output_dir (str): Directory to save the file (default: "../Data/cleaned")
        index (bool): Whether to save the DataFrame index (default: False)
    
    Returns:
        str: Full path to the saved file
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty
        OSError: If directory creation or file saving fails
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Create full file path
        file_path = os.path.join(output_dir, filename)
        
        # Save the DataFrame
        df.to_csv(file_path, index=index)
        print(f"Saved cleaned data to: {file_path}")
        
        return file_path
        
    except Exception as e:
        raise RuntimeError(f"Error saving data: {str(e)}")