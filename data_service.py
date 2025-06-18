import pandas as pd
import numpy as np
from functools import cached_property
from typing import List, Dict
import os

class ParquetDataService:
    def __init__(self, price_path="data/prices_pivot.parquet", dividend_path="data/dividends_pivot.parquet"):
        self.price_path = price_path
        self.dividend_path = dividend_path
        self._prices = None
        self._dividends = None

    @cached_property
    def get_tickers(self):
        return pd.read_parquet(
            'data/available_tickers.parquet',
            columns=['Ticker']
        )['Ticker'].dropna().astype(str).str.upper().unique().tolist()
    
    @cached_property
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
    
    def _prepare_price_data(self, tickers, start_date, end_date):
        prices = self.get_prices(tickers, start_date, end_date)
        dividends = self.get_dividends(tickers, start_date, end_date).fillna(0)

        if prices.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0)
        prev_prices = prices.shift(1)
        returns = ((prices - prev_prices + dividends) / prev_prices).fillna(0)
        return prices, dividends, returns
    
    def _get_rebalance_dates(self, index, rebalance):
        if rebalance == 'monthly':
            return index.to_series().resample('M').first().index
        elif rebalance == 'weekly':
            return index.to_series().resample('W-FRI').first().dropna().index.tolist()
        elif rebalance == 'quarterly':
            return index.to_series().resample('Q').first().index
        return [index[0]]
    
    def compute_portfolio_nav(self, weights: Dict[str, float], start_date: str, end_date: str, rebalance: str = 'monthly') -> pd.Series:
        tickers = list(weights.keys())
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Input weights must sum to 1.0")

        prices, _, returns = self._prepare_price_data(tickers, start_date, end_date)
        if prices.empty:
            return pd.Series(dtype=float)

        rebalance_dates = set(self._get_rebalance_dates(prices.index, rebalance))

        nav = [100.0]
        current_weights = np.array([weights[t] for t in tickers])
        asset_values = current_weights * 100

        for i in range(1, len(prices)):
            date = prices.index[i]
            asset_values *= (1 + returns.iloc[i])
            total_value = asset_values.sum()
            nav.append(total_value)

            if date in rebalance_dates:
                asset_values = np.array([weights[t] * total_value for t in tickers])

        return pd.Series(nav, index=prices.index)


    def compute_portfolio_weights(self, weights: Dict[str, float], start_date: str, end_date: str, rebalance: str = 'monthly') -> pd.DataFrame:
        tickers = list(weights.keys())
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Input weights must sum to 1.0")

        prices, _, returns = self._prepare_price_data(tickers, start_date, end_date)
        if prices.empty:
            return pd.DataFrame()

        rebalance_dates = set(self._get_rebalance_dates(prices.index, rebalance))

        weights_records = []
        current_weights = np.array([weights[t] for t in tickers])
        asset_values = current_weights * 100

        for i in range(1, len(prices)):
            date = prices.index[i]
            asset_values *= (1 + returns.iloc[i])
            total_value = asset_values.sum()

            if date in rebalance_dates:
                before = asset_values / total_value
                weights_records.append({
                    'date': date,
                    'rebalance_phase': 'before',
                    **{tickers[j]: before[j] for j in range(len(tickers))}
                })

                # Rebalance
                asset_values = np.array([weights[t] * total_value for t in tickers])
                after = asset_values / total_value
                weights_records.append({
                    'date': date,
                    'rebalance_phase': 'after',
                    **{tickers[j]: after[j] for j in range(len(tickers))}
                })

        return pd.DataFrame(weights_records)
    
    def _compute_for_strategies(self, strategies, start_date, end_date, rebalance, compute_fn):
        results = {}
        for name, wts in strategies.items():
            results[name] = compute_fn(
                weights=wts,
                start_date=start_date,
                end_date=end_date,
                rebalance=rebalance
            )
        return results

    def compute_navs_for_strategies(self, strategies, start_date, end_date, rebalance):
        return self._compute_for_strategies(strategies, start_date, end_date, rebalance, self.compute_portfolio_nav)

    def compute_weights_for_strategies(self, strategies, start_date, end_date, rebalance):
        return self._compute_for_strategies(strategies, start_date, end_date, rebalance, self.compute_portfolio_weights)