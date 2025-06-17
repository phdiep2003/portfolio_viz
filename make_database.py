import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm


def get_sp500_tickers():
    """Get current S&P 500 constituents from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Disable SSL verification (not recommended for production)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    table = pd.read_html(url)
    sp500_df = table[0]
    return sp500_df['Symbol'].tolist()

def download_clean_prices(tickers, start='2015-01-01', end='2024-12-31', batch_size=50):
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1} ({len(batch)} tickers)...")
        try:
            raw = yf.download(batch, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
            # Handle multi-index columns if multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw.xs('Close', axis=1, level='Price')
            else:
                # Single ticker, just one column
                close_df = raw['Close'].to_frame()
                close_df.columns = [batch[0]]
            df = close_df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Close')
            df.dropna(subset=['Close'], inplace=True)
            all_data.append(df)
        except Exception as e:
            print(f"Error downloading batch starting at index {i}: {e}")
        time.sleep(1)  # delay between batches

    if all_data:
        return pd.concat(all_data).reset_index(drop=True)
    else:
        return pd.DataFrame()

def download_dividend(tickers, start='2015-01-01', end='2024-12-31'):
    all_dividends = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            div = ticker.dividends
            if div.empty:
                continue
            div = div.reset_index()
            div.columns = ['Ex Date', 'Dividend']
            div['Ex Date'] = div['Ex Date'].dt.tz_localize(None)
            div = div[(div['Ex Date'] >= start_dt) & (div['Ex Date'] <= end_dt)]
            if div.empty:
                continue
            div['Ticker'] = symbol
            all_dividends.append(div)
        except Exception as e:
            print(f"Failed to download dividends for {symbol}: {e}")
    if all_dividends:
        full_div_df = pd.concat(all_dividends)
        full_div_df.rename(columns={'Ex Date': 'ExDate', 'Dividend': 'Dividend'}, inplace=True)
        full_div_df = full_div_df.sort_values(['Ticker', 'ExDate']).reset_index(drop=True)
        return full_div_df
    else:
        return pd.DataFrame(columns=['ExDate', 'Dividend', 'Ticker'])

def save_parquet(df, path):
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")
def merge_with_existing(new_data, existing_path, columns):
    """Merge new data with existing parquet file"""
    try:
        existing = pd.read_parquet(existing_path)
        merged = pd.concat([existing, new_data])
        # Remove duplicates (same Date+Ticker combination)
        merged = merged.drop_duplicates(columns, keep='last')
        return merged.sort_values(columns)
    except FileNotFoundError:
        return new_data

def get_sector(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('sector', 'N/A')
    except Exception as e:
        print(f"Error fetching sector for {ticker}: {e}")
        return 'Error'

def map_tickers_to_sectors(csv_path):
    """
    Map tickers to their sectors with progress display
    
    Args:
        csv_path (str): Path to CSV file containing tickers
        
    Returns:
        pd.DataFrame: DataFrame with tickers and their sectors
    """
    # Read CSV file
    tickers_df = pd.read_csv(csv_path)
    tickers_df.rename(columns={'0': 'Ticker'}, inplace=True)
    
    if 'Ticker' not in tickers_df.columns:
        raise ValueError("CSV must contain a column named 'Ticker'")
    
    # Initialize progress bar
    print("\nFetching sector information for tickers:")
    tqdm.pandas(desc="Processing tickers")
    
    # Apply with progress bar
    tickers_df['Sector'] = tickers_df['Ticker'].progress_apply(get_sector)
    
    # Print completion message
    print("\n✓ Sector mapping completed!")
    
    return tickers_df

if __name__ == "__main__":
    # === CONFIG ===
    # with open('tickers.txt', 'r') as f:
    #     tickers = [line.strip() for line in f if line.strip()]

    # print(f"Loaded {len(tickers)} tickers")

    # tickers=['BXSL','OBDC','CMG','AEM','FSK']
    # start_date = '2015-01-01'
    # end_date = datetime.today().strftime('%Y-%m-%d')
    # price_df = download_clean_prices(tickers, start=start_date, end=end_date, batch_size=50)
    # print(f"Downloaded {len(price_df)} price records")
    # price_df_load = pd.read_parquet('data/prices.parquet')
    # unique_tickers = pd.DataFrame(price_df_load['Ticker'].unique())
    # unique_tickers.to_csv('tickers.csv',index=False)
    # price_df = price_df.merge(price_df_load, on=['Date', 'Ticker'], how='outer', suffixes=('', '_old'))
    # save_parquet(price_df, path='prices.parquet')

    # dividend_df = download_dividend(tickers, start=start_date, end=end_date)
    # dividend_df_load = pd.read_parquet('data/dividends.parquet')
    # print(dividend_df_load['Ticker'].nunique())
    # missing O, OXY, HUM, JNJ, PFE, SNOW, KO, MCD 
    # tickers = ['SNOW','QBTS','IONQ']
    
    # Set date range
    # start_date = '2015-01-01'
    # end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Download prices
    # new_prices = download_clean_prices(tickers, start=start_date, end=end_date)
    # print(f"Downloaded {len(new_prices)} price records")
    
    # # Merge with existing data
    # final_prices = merge_with_existing(new_prices, 'data/prices.parquet',['Date', 'Ticker'])
    
    # # Save results
    # save_parquet(final_prices, 'prices.parquet')
    
    # # Show summary
    # print(f"Final dataset contains {len(final_prices)} price records")
    # print(f"Unique tickers in final dataset: {final_prices['Ticker'].nunique()}")
    # new_div = download_dividend(tickers, start=start_date, end=end_date)
    # final_div = merge_with_existing(new_div, 'data/dividends.parquet',['ExDate', 'Ticker'])
    # final_div.to_parquet('dividends.parquet', index=False)
    # print(f"Dividends: {len(final_div)} records | {final_div['Ticker'].nunique()} tickers")
    # print(final_div.head())

    df_with_sectors = map_tickers_to_sectors('tickers.csv')
    df_with_sectors.to_csv('tickers_with_sectors.csv', index=False)
    print(df_with_sectors.head())

    # for each ticker in tickers, find the corresponding sectors on yfinance


