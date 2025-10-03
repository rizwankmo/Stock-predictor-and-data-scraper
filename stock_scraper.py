import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse

def download_stock_data(ticker, start_date=None, end_date=None, period=None):
    """
    Download historical stock data using Yahoo Finance API
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date (str): Start date in 'YYYY-MM-DD' format. Default: 5 years ago.
    end_date (str): End date in 'YYYY-MM-DD' format. Default: today.
    period (str): Alternative to specifying dates. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'
    
    Returns:
    pandas.DataFrame: Stock data including Open, High, Low, Close, Volume, and Dividends
    """
    today = datetime.now()
    
    if period:
        data = yf.download(ticker, period=period)
    else:
        if not start_date:
            start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = today.strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date)
    
    return data

def save_data(data, ticker, output_format='csv'):
    """
    Save downloaded data to a file
    
    Parameters:
    data (pandas.DataFrame): Stock data to save
    ticker (str): Stock ticker symbol
    output_format (str): Output file format ('csv' or 'excel')
    
    Returns:
    str: Path to the saved file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{ticker}_{timestamp}"
    
    if output_format.lower() == 'csv':
        filepath = f"{filename}.csv"
        data.to_csv(filepath)
    elif output_format.lower() in ['excel', 'xlsx']:
        filepath = f"{filename}.xlsx"
        data.to_excel(filepath)
    else:
        raise ValueError("Output format must be 'csv' or 'excel'")
    
    return filepath

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Download historical stock data')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', help='Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)')
    parser.add_argument('--format', default='csv', help='Output file format (csv or excel)')
    args = parser.parse_args()
    
    try:
        # Download data
        print(f"Downloading data for {args.ticker}...")
        data = download_stock_data(args.ticker, args.start, args.end, args.period)
        
        # Save data
        filepath = save_data(data, args.ticker, args.format)
        print(f"Data saved to {filepath}")
        
        # Display data summary
        print("\nData Summary:")
        print(f"Rows: {len(data)}")
        print(f"Date Range: {data.index.min()} to {data.index.max()}")
        print("\nFirst few rows:")
        print(data.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()