import pandas as pd

def load_data(
    file_path, 
    companies=None,
    parse_dates=True,
    date_column='date',
    coerce_dates=True
):
    """
    Load data from a CSV file with flexible date parsing options.
    
    Args:
        file_path (str): Path to the CSV file
        companies (List[str], optional): List of company symbols to filter by
        parse_dates (bool): Whether to parse dates (default: True)
        date_column (str): Name of the date column (default: 'date')
        coerce_dates (bool): Whether to use coerce for date parsing. 
                           Set True for news data with potential inconsistent dates,
                           False for clean data like stock prices (default: False)
    
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Parse dates if requested
        if parse_dates and date_column in df.columns:
            # Try parsing with a specific format first
            df[date_column] = pd.to_datetime(
                df[date_column],
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )
            
            # If any dates failed, they might be in timezone format
            mask = df[date_column].isna()
            if mask.any():
                # Try parsing the failed dates with flexible parser
                df.loc[mask, date_column] = pd.to_datetime(
                    df.loc[mask, date_column],
                    errors='coerce',
                    format=None  # Let pandas detect the format
                )
            
            # If coerce_dates is False and we have any NaT values, raise an error
            if not coerce_dates and df[date_column].isna().any():
                raise ValueError("Invalid date format found in data")
            
        # Filter by companies if specified
        if companies is not None and 'stock' in df.columns:
            df = df[df['stock'].isin(companies)]
            
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise