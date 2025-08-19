import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from functools import cached_property
from typing import List, Dict
from numba import njit

@njit
def nav_jit_optimized(returns: np.ndarray, weights: np.ndarray, rebalance_flags: np.ndarray) -> np.ndarray:
    """
    Computes portfolio NAV using vectorized NumPy operations inside the loop for maximum speed.
    """
    T = returns.shape[0]
    
    # Specify the data type for precision in financial calculations
    nav = np.empty(T, dtype=np.float64)
    nav[0] = 100.0
    
    asset_vals = (weights * 100.0).astype(np.float64)

    for t in range(1, T):
        # 1. Daily portfolio growth (already efficient)
        asset_vals *= (1 + returns[t])
        
        # 2. OPTIMIZED: Replaced a manual loop with a fast NumPy sum
        total_val = np.sum(asset_vals)
        nav[t] = total_val

        # 3. OPTIMIZED: Replaced a manual loop with a vectorized multiplication
        if rebalance_flags[t]:
            asset_vals = weights * total_val
            
    return nav

class ParquetDataService:
    def __init__(self, price_path="data/prices_partitioned", dividend_path="data/dividends_partitioned"):
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
    
    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Optimized to use set_index/unstack instead of pivot and reindex instead of a loop.
        """
        dataset = ds.dataset(self.price_path, format='parquet', partitioning='hive')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        filter_expr = (
            ds.field('Ticker').isin(tickers) &
            (ds.field('Date') >= start_dt) &
            (ds.field('Date') <= end_dt)
        )

        # Explicitly select only necessary columns to minimize memory usage
        table = dataset.to_table(filter=filter_expr, columns=['Date', 'Ticker', 'Price'])
        df = table.to_pandas()

        if df.empty or len(df) < 10:
            raise ValueError(
                "Available data range is 2015–2025. "
                "Need more? Please hire me as your investment analyst!"
                "hungphatdiep03@gmail.com +974 3063-6181"
            )

        # This is often faster and more memory-efficient than `pivot`.
        df_wide = df.set_index(['Date', 'Ticker'])['Price'].unstack()
        df_wide = df_wide.reindex(columns=tickers)

        # Validation check remains the same
        last_row = df_wide.iloc[-1]
        missing_at_end = last_row[last_row.isna()].index.tolist()

        if missing_at_end:
            raise ValueError(f"The following tickers have no data at the end of the selected range: {missing_at_end}")

        return df_wide

    def get_dividends(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Optimized to avoid pivot and manual column creation. Uses set_index and reindex.
        """
        dataset = ds.dataset(self.dividend_path, format='parquet', partitioning='hive')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        filter_expr = (
            ds.field('Ticker').isin(tickers) &
            (ds.field('Date') >= start_dt) &
            (ds.field('Date') <= end_dt)
        )
        
        # Select only the columns you need *before* loading into memory
        table = dataset.to_table(filter=filter_expr, columns=['Date', 'Ticker', 'Price'])
        df = table.to_pandas()

        if df.empty:
            # Create an empty DataFrame with the correct structure to avoid downstream errors
            return pd.DataFrame(columns=tickers)

        # Use set_index and unstack, which is often more efficient than pivot
        df = df.set_index(['Date', 'Ticker'])['Price'].unstack()

        # Use reindex to add missing tickers and ensure correct column order in one step
        df = df.reindex(columns=tickers, fill_value=np.nan)

        return df


    def get_returns(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Optimized to use vectorized operations for reinvestment and return calculations.
        """
        prices = self.get_prices(tickers, start_date, end_date)
        dividends = self.get_dividends(tickers, start_date, end_date)
        
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame()

        start_dt = prices.index[0]
        end_dt = prices.index[-1]
        
        # Directly use pandas indexing which is clear and efficient
        start_prices = prices.ffill().bfill().iloc[0]
        end_prices = prices.ffill().iloc[-1]

        # Calculate reinvested dividends using vectorized operations
        if not dividends.empty:
            # Align dividends with the requested tickers
            divs = dividends.reindex(columns=tickers).fillna(0)
            
            # Vectorized calculation of growth factors
            days_to_end = (end_dt - divs.index.to_series()).dt.days.values[:, np.newaxis]
            years_to_end = days_to_end / 365.25
            growth_factors = np.power(1.04, years_to_end)
            
            reinvested_sum = pd.Series(np.sum(divs.values * growth_factors, axis=0), index=tickers)
        else:
            reinvested_sum = pd.Series(0.0, index=tickers)

        # Assemble the final DataFrame using aligned Series
        df = pd.DataFrame(index=tickers)
        df['Years'] = (end_dt - start_dt).days / 365.25
        df['Close_Start'] = start_prices
        df['Close_End'] = end_prices
        df['Reinvested'] = reinvested_sum
        
        # Fully vectorized return calculations
        df['Total Return'] = (df['Close_End'] + df['Reinvested']) / df['Close_Start'] - 1
        df['Annualized Return'] = (1 + df['Total Return']) ** (1 / df['Years']) - 1

        return df[['Years', 'Total Return', 'Annualized Return']], prices


    def _prepare_price_data(self, tickers, start_date, end_date):
        """
        Optimized to fetch dividends only once and perform efficient alignment.
        """
        prices = self.get_prices(tickers, start_date, end_date)
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Pass the full date range of prices to get relevant dividends
        div_start = prices.index.min().strftime('%Y-%m-%d')
        div_end = prices.index.max().strftime('%Y-%m-%d')
        dividends = self.get_dividends(tickers, div_start, div_end)

        # Align dividends to the prices DataFrame in one step.
        aligned_dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0)

        # Vectorized calculation for total return series
        prev_prices = prices.shift(1)
        returns = (prices - prev_prices + aligned_dividends) / prev_prices
        
        # Fill any NaNs resulting from the shift operation at the beginning
        returns.iloc[0] = 0.0

        return prices, aligned_dividends, returns
   
    def _get_rebalance_dates(self, index: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
        """
        Calculates rebalance dates.
        """
        if rebalance == 'monthly':
            return index.to_series().resample('ME').first().index
        elif rebalance == 'weekly':
            # OPTIMIZED: Removed .tolist() to avoid an unnecessary type conversion.
            return index.to_series().resample('W-FRI').first().dropna().index
        elif rebalance == 'quarterly':
            return index.to_series().resample('Q').first().index
    
    def compute_portfolio_nav(self, weights: Dict[str, float], start_date: str, end_date: str, rebalance: str = 'monthly') -> pd.Series:
        """
        Calculates portfolio NAV, calling the optimized JIT function.
        """
        tickers = list(weights.keys())
        
        # NOTE: The performance of _prepare_price_data is a likely bottleneck and should be profiled.
        prices, _, returns = self._prepare_price_data(tickers, start_date, end_date)
        if prices.empty:
            return pd.Series(dtype=float)

        # This section efficiently creates the boolean flags for rebalancing days
        rebalance_dates = self._get_rebalance_dates(prices.index, rebalance)
        rebalance_indices = prices.index.get_indexer(rebalance_dates, method='pad')
        rebalance_indices = rebalance_indices[rebalance_indices != -1]
        
        rebalance_flags = np.zeros(len(prices), dtype=bool)
        rebalance_flags[rebalance_indices] = True

        # Prepare NumPy arrays for the JIT function
        weights_vec = np.array([weights[t] for t in tickers])
        returns_arr = returns[tickers].values

        # Call the fully optimized Numba function
        nav_arr = nav_jit_optimized(returns_arr, weights_vec, rebalance_flags)
        
        return pd.Series(nav_arr, index=prices.index)


    # def compute_portfolio_weights(self, weights: Dict[str, float], start_date: str, end_date: str, rebalance: str = 'monthly') -> pd.DataFrame:
    #     tickers = list(weights.keys())
    #     if not np.isclose(sum(weights.values()), 1.0):
    #         raise ValueError("Input weights must sum to 1.0")

    #     prices, _, returns = self._prepare_price_data(tickers, start_date, end_date)
    #     if prices.empty:
    #         return pd.DataFrame()

    #     rebalance_dates = set(self._get_rebalance_dates(prices.index, rebalance))

    #     records = []
    #     current_weights = np.array([weights[t] for t in tickers])
    #     asset_values = current_weights * 100

    #     for i in range(1, len(prices)):
    #         date = prices.index[i]
    #         asset_values *= (1 + returns.iloc[i])
    #         total_value = asset_values.sum()

    #         if date in rebalance_dates:
    #             before_weights = asset_values / total_value
    #             after_weights = np.array([weights[t] for t in tickers])

    #             records.append([date, 'before', *before_weights])
    #             # rebalance asset values
    #             asset_values = after_weights * total_value
    #             records.append([date, 'after', *after_weights])

    #     # Build DataFrame once at the end
    #     columns = ['date', 'rebalance_phase'] + tickers
    #     df = pd.DataFrame(records, columns=columns)

    #     return df
    
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

    # def compute_weights_for_strategies(self, strategies, start_date, end_date, rebalance):
    #     return self._compute_for_strategies(strategies, start_date, end_date, rebalance, self.compute_portfolio_weights)