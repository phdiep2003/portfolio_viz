import pandas as pd
import numpy as np
from functools import lru_cache
from typing import List, Dict
import os

class ParquetDataService:
    def __init__(self, price_path="data/prices_pivot.parquet", dividend_path="data/dividends_pivot.parquet"):
        self.price_path = price_path
        self.dividend_path = dividend_path
        self._prices = None
        self._dividends = None

    @property
    @lru_cache(maxsize=1)
    def get_tickers(self):
        return sorted(self._load_prices().columns.tolist())
    
    @property
    @lru_cache(maxsize=1)
    def tickers_with_sectors(self):
        return pd.read_parquet('data/tickers_with_sectors.parquet')
    
    def _load_prices(self) -> pd.DataFrame:
        if self._prices is None:
            self._prices = pd.read_parquet(self.price_path)
            self._prices.index = pd.to_datetime(self._prices.index)
            self._prices.sort_index(inplace=True)
        return self._prices

    def _load_dividends(self) -> pd.DataFrame:
        if self._dividends is None and os.path.exists(self.dividend_path):
            df = pd.read_parquet(self.dividend_path)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            self._dividends = df
        return self._dividends if self._dividends is not None else pd.DataFrame()

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_prices()
        return df.loc[start_date:end_date, tickers]

    def get_dividends(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_dividends()
        if df.empty:
            return pd.DataFrame(columns=tickers)

        existing_tickers = [t for t in tickers if t in df.columns]
        df = df.loc[start_date:end_date, existing_tickers]

        missing_tickers = [t for t in tickers if t not in existing_tickers]
        for t in missing_tickers: # Add missing tickers as columns with NaN
            df[t] = np.nan
        
        df = df[tickers] # Reorder columns to match requested tickers

        return df

    def get_returns(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        prices = self.get_prices(tickers, start_date, end_date)
        dividends = self.get_dividends(tickers, start_date, end_date)
        if prices.empty:
            return pd.DataFrame()

        end_dt = pd.to_datetime(end_date)
        start_prices = prices.ffill().iloc[0]
        end_prices = prices.ffill().iloc[-1]

        df = pd.DataFrame({
            'Ticker': tickers,
            'Close_Start': start_prices[tickers].values,
            'Close_End': end_prices[tickers].values
        })

        if not dividends.empty:
            existing_tickers = [t for t in tickers if t in dividends.columns]
            if existing_tickers:
                divs = dividends[existing_tickers].copy()
                divs = divs.loc[~divs.index.isna()]  # Clean dates

                days = (end_dt - divs.index.to_series()).dt.days.values[:, None]
                years = days / 365.25
                growth_factors = np.power(1.04, years)

                reinvested_values = divs.values * growth_factors
                reinvested = pd.DataFrame(reinvested_values, index=divs.index, columns=divs.columns)
                reinvested_sum = reinvested.sum()

                reinvested_sum_full = pd.Series(0, index=tickers)
                reinvested_sum_full.update(reinvested_sum)
            else:
                reinvested_sum_full = pd.Series(0, index=tickers)
        else:
            reinvested_sum_full = pd.Series(0, index=tickers)
        df['Reinvested'] = df['Ticker'].map(reinvested_sum_full.to_dict())

        df['Years'] = (end_dt - prices.index[0]).days / 365.25
        df['Total Return'] = ((df['Close_End'] + df['Reinvested']) / df['Close_Start']) - 1
        df['Annualized Return'] = (1 + df['Total Return']) ** (1 / df['Years']) - 1
        df.set_index('Ticker',inplace=True)
        return df[['Years', 'Total Return', 'Annualized Return']]

    def filter_date_range(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        return df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    def compute_portfolio_value(self, weights: Dict[str, float], start_date: str, end_date: str, rebalance: str = 'monthly') -> pd.Series:
        # Get tickers and validate weights
        tickers = list(weights.keys())
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Input weights must sum to 1.0")

        # Get price and dividend data
        prices_pivot = self.get_prices(tickers, start_date, end_date)
        dividends_pivot = self.get_dividends(tickers, start_date, end_date).fillna(0)

        if prices_pivot.empty:
            return pd.Series(dtype=float)

        # Align dividend dates with price dates
        dividends_aligned = dividends_pivot.reindex(index=prices_pivot.index, columns=prices_pivot.columns).fillna(0)

        # Calculate daily returns (price + dividend)
        prev_prices = prices_pivot.shift(1)
        returns = ((prices_pivot - prev_prices + dividends_aligned) / prev_prices).fillna(0)

        # Determine rebalance dates
        if rebalance == 'monthly':
            rebalance_dates = prices_pivot.resample('M').first().index
        elif rebalance == 'weekly':
            rebalance_dates = prices_pivot.resample('W').first().index
        elif rebalance == 'quarterly':
            rebalance_dates = prices_pivot.resample('Q').first().index
        else:  # no rebalancing
            rebalance_dates = [prices_pivot.index[0]]

        # Initialize portfolio tracking
        nav = [100.0]  # Starting value normalized to 100
        current_weights = np.array([weights[t] for t in tickers])  # Current allocation weights
        asset_values = current_weights * 100  # Dollar value in each asset

        for i in range(1, len(prices_pivot)):
            date = prices_pivot.index[i]
            
            # Update asset values with today's returns
            asset_values *= (1 + returns.iloc[i])
            current_portfolio_value = asset_values.sum()
            nav.append(current_portfolio_value)
            
            # Handle rebalancing
            if date in rebalance_dates:
                # Calculate actual weights before rebalancing (for debugging)
                # actual_weights = asset_values / current_portfolio_value
                # print(f"\nRebalancing at {date.date()}")
                # print("Pre-rebalance weights:", {t: f"{w:.2%}" for t, w in zip(tickers, actual_weights)})
                # print("Target weights:", {t: f"{w:.2%}" for t, w in weights.items()})
                
                # Rebalance to target weights
                asset_values = np.array([weights[t] * current_portfolio_value for t in tickers])
            
            # Update current weights for next iteration
            current_weights = asset_values / current_portfolio_value

        return pd.Series(nav, index=prices_pivot.index)