import pandas as pd
from functools import lru_cache
from typing import List
import os

# --- ParquetDataService with Caching ---
class ParquetDataService:
    def __init__(self, price_path="data/prices.parquet", dividend_path="data/dividends.parquet"):
        self.price_path = price_path
        self.dividend_path = dividend_path
        self._prices = None
        self._dividends = None

    @lru_cache(maxsize=1)
    def get_tickers(self):
        prices = self._load_prices()
        return sorted(prices['Ticker'].unique())
    
    @property
    @lru_cache(maxsize=1)
    def tickers_with_sectors(self):
        if not hasattr(self, '_sectors'):
            self._sectors = pd.read_parquet('data/tickers_with_sectors.parquet')
        return self._sectors
    
    def _load_prices(self) -> pd.DataFrame:
        if self._prices is None:
            self._prices = pd.read_parquet(self.price_path)
            self._prices['Date'] = pd.to_datetime(self._prices['Date'], errors='coerce')
        return self._prices

    def _load_dividends(self) -> pd.DataFrame:
        if self._dividends is None and os.path.exists(self.dividend_path):
            df = pd.read_parquet(self.dividend_path)
            df['ExDate'] = pd.to_datetime(df['ExDate'], errors='coerce')
            self._dividends = df
        return self._dividends if self._dividends is not None else pd.DataFrame()

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_prices()
        df = df[df['Ticker'].isin(tickers)]
        return self.filter_date_range(df, 'Date', start_date, end_date)

    def get_dividends(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_dividends()
        if df.empty:
            return df
        df = df[df['Ticker'].isin(tickers)]
        return self.filter_date_range(df, 'ExDate', start_date, end_date)

    def get_returns(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        prices = self.get_prices(tickers, start_date, end_date)
        dividends = self.get_dividends(tickers, start_date, end_date)
        if prices.empty:
            return pd.DataFrame()

        end_dt = pd.to_datetime(end_date)

        # Get first and last price per ticker
        start_prices = prices.sort_values('Date').groupby('Ticker').first().reset_index()
        end_prices = prices.sort_values('Date').groupby('Ticker').last().reset_index()

        start = start_prices[['Ticker', 'Date', 'Close']].rename(columns={'Date': 'Date_Start', 'Close': 'Close_Start'})
        end = end_prices[['Ticker', 'Close']].rename(columns={'Close': 'Close_End'})
        merged = pd.merge(start, end, on='Ticker')

        # Reinvested dividends
        if not dividends.empty:
            dividends = pd.merge(dividends, start_prices[['Ticker', 'Date']], on='Ticker', suffixes=('', '_Start'))
            dividends['Days'] = (end_dt - dividends['ExDate']).dt.days
            dividends['Years'] = dividends['Days'] / 365.25
            dividends['Reinvested'] = dividends['Dividend'] * (1.04 ** dividends['Years'])  # reinvest at 4% annually
            reinvested = dividends.groupby('Ticker')['Reinvested'].sum().reset_index()
        else:
            reinvested = pd.DataFrame({'Ticker': tickers, 'Reinvested': [0] * len(tickers)})

        merged = pd.merge(merged, reinvested, on='Ticker', how='left')
        merged['Reinvested'] = merged['Reinvested'].fillna(0)
        merged['Years'] = (end_dt - merged['Date_Start']).dt.days / 365.25
        merged['Total Return'] = ((merged['Close_End'] + merged['Reinvested']) / merged['Close_Start']) - 1
        merged['Annualized Return'] = (1 + merged['Total Return']) ** (1 / merged['Years']) - 1

        return merged[['Ticker', 'Years', 'Total Return', 'Annualized Return']]

    def filter_date_range(self, df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        return df[(df[date_col] >= start) & (df[date_col] <= end)]

    def process_dividends(self, dividend_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        if dividend_df.empty or price_df.empty:
            return pd.DataFrame(index=price_df.index if not price_df.empty else [])

        grouped = dividend_df.groupby(['Ticker', 'ExDate'])['Dividend'].sum().reset_index()
        pivoted = grouped.pivot(index='ExDate', columns='Ticker', values='Dividend')
        return pivoted.reindex(price_df.index).fillna(0)
