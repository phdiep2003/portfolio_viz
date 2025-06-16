import yfinance as yf
import pandas as pd
from datetime import datetime
import time


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
        full_div_df = full_div_df.sort_values(['Ticker', 'Ex Date']).reset_index(drop=True)
        return full_div_df
    else:
        return pd.DataFrame(columns=['Ex Date', 'Dividend', 'Ticker'])

def save_parquet(df, path):
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")

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
    # price_df = price_df.merge(price_df_load, on=['Date', 'Ticker'], how='outer', suffixes=('', '_old'))
    # save_parquet(price_df, path='prices.parquet')

    # dividend_df = download_dividend(tickers, start=start_date, end=end_date)
    # dividend_df_load = pd.read_parquet('data/dividends.parquet')
    # print(f"Downloaded {len(dividend_df)} dividend records")
    # dividend_df = dividend_df.merge(dividend_df_load, on=['Ex Date', 'Ticker'], how='outer', suffixes=('', '_old'))
    # save_parquet(dividend_df, path='dividends.parquet')
    # Load dividend data
    df = pd.read_parquet('data/prices.parquet')
    print(f"Loaded {len(df)} price records")
    # Ensure Date is datetime and sorted
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = df['Close'].combine_first(df['Close_old'])

    # Drop the extra column
    df.drop(columns=['Close_old'], inplace=True)
    df = df.dropna(subset=['Date', 'Ticker'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['Ticker', 'Date'])

    # Save cleaned file
    df.to_parquet('prices.parquet', index=False)
    print(df.columns)
    print(f"Cleaned price data saved. Rows: {len(df)}, Tickers: {df['Ticker'].nunique()}")

    div = pd.read_parquet('data/dividends.parquet')
    print(f"Loaded {len(div)} dividend records")
    # Standardize columns
    div.rename(columns={'Ex Date': 'ExDate'}, inplace=True)

    # Convert to datetime and drop invalid
    div['ExDate'] = pd.to_datetime(div['ExDate'], errors='coerce')
    div['Dividend'] = div['Dividend'].combine_first(div['Dividend_old'])

    # Drop the extra column
    div.drop(columns=['Dividend_old'], inplace=True)

    div = div.dropna(subset=['ExDate', 'Ticker'])
    # Remove duplicates
    div = div.drop_duplicates(subset=['Ticker', 'ExDate'])
    # Sort
    div = div.sort_values(['Ticker', 'ExDate'])
    # Save cleaned file
    div.to_parquet('dividends.parquet', index=False)
    print(div.columns)
    print(f"Cleaned dividend data saved. Rows: {len(div)}, Tickers: {div['Ticker'].nunique()}")

# missing O, OXY, HUM, JNJ, PFE