from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json, pickle
import os
from functools import wraps

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "123459876"  # TODO: Replace with secure secret key in production

prices_df = pd.read_parquet('data/prices.parquet')

# Assuming the ticker column is named 'ticker' or similar
unique_tickers = sorted(prices_df['Ticker'].unique())

# --- Configuration ---
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

class Config:
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2025-01-01"
    RISK_FREE_RATE = 0.03
    TARGET_RETURN = 0.15
    TARGET_RISK = 0.13
    CACHE_EXPIRY_DAYS = 10  # Number of days to keep cached data

# --- Cache Utilities ---
def make_cache_key(selected_tickers, start_date, end_date, target_return, target_volatility, min_weights, max_weights):
    # Sort tickers and weights to ensure deterministic key
    key_dict = {
        "tickers": sorted(selected_tickers),
        "start_date": start_date,
        "end_date": end_date,
        "target_return": target_return,
        "target_volatility": target_volatility,
        "min_weights": {k: min_weights[k] for k in sorted(min_weights)},
        "max_weights": {k: max_weights[k] for k in sorted(max_weights)}
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def save_cache(key: str, data: Any) -> None:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cache(key: str) -> Any:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def cache_exists(key: str) -> bool:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if not os.path.exists(cache_path):
        return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return (datetime.now() - cache_time).days < Config.CACHE_EXPIRY_DAYS

# --- Data Classes ---
@dataclass
class OptimizationResult:
    strategy: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float

class PortfolioOptimizer:
    def compute_corr_matrix(prices: pd.DataFrame, cache_key_hint: str) -> pd.DataFrame:
        """Compute the correlation matrix for asset returns."""
        return prices.pct_change().dropna().corr()
    
    @staticmethod
    def run_optimizations(
        mu: pd.Series, 
        prices: pd.DataFrame, 
        bounds: List[Tuple[float, float]],
        risk_free_rate: float,
        target_return: float,
        target_volatility: float,
    ) -> Dict:
        """Run portfolio optimization strategies."""
        prices = prices.dropna()
        daily_returns = prices.pct_change().dropna()
        annualized_vol = daily_returns.std() * (252 ** 0.5)
        annualized_vol = annualized_vol.reindex(mu.index, fill_value=0)
        cov_matrix = CovarianceShrinkage(prices).ledoit_wolf()

        per_asset_metrics = pd.DataFrame({
            'Ticker': mu.index,
            'Expected Return': mu,
            'Volatility': annualized_vol,
            'Sharpe Ratio': (mu - risk_free_rate) / annualized_vol
        })
        results = {}

        strategies = {
            "Min Volatility": lambda ef: ef.min_volatility(),
            "Max Sharpe": lambda ef: ef.max_sharpe(risk_free_rate=risk_free_rate),
            f"Efficient Return {target_return:.3f}": lambda ef: ef.efficient_return(target_return),
            f"Efficient Risk {target_volatility:.3f}": lambda ef: ef.efficient_risk(target_volatility),
        }

        for name, strategy in strategies.items():
            # Generate cache key for this strategy
            try:
                mu = mu.astype("float64")
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=bounds)
                strategy(ef)
                performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                # Create the result object
                result = OptimizationResult(
                    strategy=name,
                    weights=ef.clean_weights(),
                    expected_return=performance[0],
                    volatility=performance[1],
                    sharpe_ratio=performance[2]
                )

                # Save result to cache
            except Exception as e:
                print(f"Error in {name} strategy: {e}")
                result = {"Error": str(e)}

            # Store the result in the results dictionary
            results[name] = result
        return results, per_asset_metrics

    @staticmethod
    def compile_results(results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split optimization results into performance summary and allocation tables."""
        perf_rows = []
        alloc_rows = []

        for strategy, data in results.items():
            if isinstance(data, dict) and "Error" in data:
                continue

            # Performance summary
            perf_rows.append({
                "Strategy": data.strategy,
                "Expected Return": data.expected_return,
                "Volatility": data.volatility,
                "Sharpe Ratio": data.sharpe_ratio
            })

            # Allocation
            alloc_row = {"Strategy": data.strategy}
            alloc_row.update(data.weights)
            alloc_rows.append(alloc_row)

        return pd.DataFrame(perf_rows), pd.DataFrame(alloc_rows)

@dataclass
class ChartData:
    ticker: str
    expected_return: float
    chart_html: str

# --- ParquetDataService with Caching ---
class ParquetDataService:
    def __init__(self, price_path="data/prices.parquet", dividend_path="data/dividends.parquet"):
        self.price_path = price_path
        self.dividend_path = dividend_path

    def get_prices(self, Tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = pd.read_parquet(self.price_path)
        df = df[df['Ticker'].isin(Tickers)]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        return df

    def get_dividends(self, Tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not os.path.exists(self.dividend_path):
            return pd.DataFrame()
        df = pd.read_parquet(self.dividend_path)
        df = df[df['Ticker'].isin(Tickers)]
        df['ExDate'] = pd.to_datetime(df['ExDate'])
        df = df[(df['ExDate'] >= start_date) & (df['ExDate'] <= end_date)]
        return df

    def get_returns(self, Tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        prices = self.get_prices(Tickers, start_date, end_date)
        dividends = self.get_dividends(Tickers, start_date, end_date)
        if prices.empty:
            return pd.DataFrame()

        end_dt = pd.to_datetime(end_date)
        start_prices = prices.sort_values('Date').groupby('Ticker').first().reset_index()
        end_prices = prices.sort_values('Date').groupby('Ticker').last().reset_index()

        start = start_prices[['Ticker', 'Date', 'Close']].rename(columns={'Date': 'Date_Start', 'Close': 'Close_Start'})
        end = end_prices[['Ticker', 'Close']].rename(columns={'Close': 'Close_End'})
        merged = pd.merge(start, end, on='Ticker')

        if not dividends.empty:
            dividends = pd.merge(dividends, start_prices[['Ticker', 'Date']], on='Ticker', suffixes=('', '_Start'))
            dividends['Days'] = (end_dt - dividends['ExDate']).dt.days
            dividends['Years'] = dividends['Days'] / 365.25
            dividends['Reinvested'] = dividends['Dividend'] * (1.04 ** dividends['Years'])
            reinvested = dividends.groupby('Ticker')['Reinvested'].sum().reset_index()
        else:
            reinvested = pd.DataFrame(columns=['Ticker', 'Reinvested'])

        merged = pd.merge(merged, reinvested, on='Ticker', how='left')
        merged['Reinvested'] = merged['Reinvested'].fillna(0)
        merged['Years'] = (end_dt - merged['Date_Start']).dt.days / 365.25
        merged['Total Return'] = ((merged['Close_End'] + merged['Reinvested']) / merged['Close_Start']) - 1
        merged['Annualized Return'] = (1 + merged['Total Return']) ** (1 / merged['Years']) - 1
        return merged[['Ticker', 'Reinvested', 'Date_Start', 'Close_Start', 'Close_End', 'Total Return', 'Annualized Return']]

    def filter_date_range(self, df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        return df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]

    def process_dividends(self, dividend_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        grouped = dividend_df.groupby(['Ticker', 'ExDate'])['Dividend'].sum().reset_index()
        pivoted = grouped.pivot(index='ExDate', columns='Ticker', values='Dividend')
        return pivoted.reindex(price_df.index).fillna(0)


@app.route('/api/tickers')
def tickers_api():
    q = request.args.get('q', '').upper()
    # assuming `all_tickers` is your master list in memory or cache
    matches = [t for t in unique_tickers if t.startswith(q)]
    return jsonify(matches[:20])  # limit to 20 results

@app.route('/', methods=['GET', 'POST'])
def unified_portfolio():
    df = pd.read_parquet('data/prices.parquet')
    tickers = sorted(df['Ticker'].unique().tolist())

    if request.method == 'POST':
        try:
            asset_count = int(request.form.get('asset_count', 10))
            selected_tickers = [request.form.get(f'tickers_{i}') for i in range(asset_count)]
            start_date = request.form.get('start_date', Config.DEFAULT_START_DATE)
            end_date = request.form.get('end_date', Config.DEFAULT_END_DATE)
            target_return = float(request.form['target_return'])
            target_volatility = float(request.form['target_volatility'])

            min_weights = {}
            max_weights = {}
            for i, ticker in enumerate(selected_tickers):
                if ticker:  # skip empty rows
                    min_value = float(request.form.get(f"min_{i}", 0.0)) / 100  # convert to decimal
                    max_value = float(request.form.get(f"max_{i}", 20.0)) / 100
                    min_weights[ticker] = min_value
                    max_weights[ticker] = max_value

            # Generate cache key
            cache_key_val = make_cache_key(
                selected_tickers, start_date, end_date,
                target_return, target_volatility,
                min_weights, max_weights
            )

            # Check if cached result exists
            if cache_exists(cache_key_val):
                cached_data = load_cache(cache_key_val)
                return render_template(
                    'unified_portfolio.html',
                    all_tickers=tickers,
                    selected_tickers=selected_tickers,
                    asset_count=len(selected_tickers),
                    start_date=start_date,
                    end_date=end_date,
                    mu=cached_data['mu'],
                    tr=cached_data['tr'],
                    df_perf=cached_data['df_perf'],
                    df_alloc=cached_data['df_alloc'],
                    corr_matrix=cached_data['corr_matrix'],
                    min_weights=min_weights,
                    max_weights=max_weights,
                    target_return=target_return,
                    target_volatility=target_volatility
                )

            # === No cache found, do full calculation ===
            data_service = ParquetDataService()

            returns_df = data_service.get_returns(selected_tickers, start_date, end_date)

            # Check for missing return data per ticker
            missing_returns = [t for t in selected_tickers if t and returns_df[returns_df['Ticker'] == t].dropna().empty]
            if missing_returns:
                error_message = f"No return data for tickers: {', '.join(missing_returns)}"
                return render_template(
                    'unified_portfolio.html',
                    selected_tickers=selected_tickers,
                    asset_count=len(selected_tickers),
                    start_date=start_date,
                    end_date=end_date,
                    mu=None,
                    tr=None,
                    df_perf=None,
                    df_alloc=None,
                    corr_matrix=None,
                    min_weights=min_weights,
                    max_weights=max_weights,
                    target_return=target_return,
                    target_volatility=target_volatility,
                    error=error_message
                )

            mu = returns_df.groupby('Ticker')['Annualized Return'].first()
            tr = returns_df.groupby('Ticker')['Total Return'].first()

            prices_df = data_service.get_prices(selected_tickers, start_date, end_date)

            # Check for missing price data per ticker
            missing_prices = [t for t in selected_tickers if t and prices_df[prices_df['Ticker'] == t].empty]
            if missing_prices:
                error_message = f"No price data for tickers: {', '.join(missing_prices)}"
                return render_template(
                    'unified_portfolio.html',
                    all_tickers=tickers,
                    selected_tickers=selected_tickers,
                    asset_count=len(selected_tickers),
                    start_date=start_date,
                    end_date=end_date,
                    mu=None,
                    tr=None,
                    df_perf=None,
                    df_alloc=None,
                    corr_matrix=None,
                    min_weights=min_weights,
                    max_weights=max_weights,
                    target_return=target_return,
                    target_volatility=target_volatility,
                    error=error_message
                )

            prices_df = data_service.filter_date_range(prices_df, 'Date', start_date, end_date)
            price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
            mu = mu.loc[price_pivot.columns]  # Ensure order matches price_pivot columns

            optimizer = PortfolioOptimizer()
            bounds = [(min_weights.get(t, 0.0), max_weights.get(t, 0.2)) for t in mu.index]

            results, per_asset_metrics = optimizer.run_optimizations(
                mu, price_pivot,
                bounds=bounds,
                risk_free_rate=Config.RISK_FREE_RATE,
                target_return=target_return,
                target_volatility=target_volatility
            )

            df_perf, df_alloc = optimizer.compile_results(results)
            corr_matrix = PortfolioOptimizer.compute_corr_matrix(price_pivot, '_'.join(sorted(price_pivot.columns)))

            # Save cache
            save_cache(cache_key_val, {
                'mu': mu,
                'tr': tr,
                'df_perf': df_perf,
                'df_alloc': df_alloc,
                'corr_matrix': corr_matrix
            })

            return render_template(
                'unified_portfolio.html',
                selected_tickers=selected_tickers,
                asset_count=len(selected_tickers),
                start_date=start_date,
                end_date=end_date,
                mu=mu,
                tr=tr,
                df_perf=df_perf,
                df_alloc=df_alloc,
                corr_matrix=corr_matrix,
                min_weights=min_weights,
                max_weights=max_weights,
                target_return=target_return,
                target_volatility=target_volatility
            )

        except Exception as e:
            return f"Internal Server Error: {e}", 500

    # GET fallback
    return render_template(
        'unified_portfolio.html',
        selected_tickers=[],
        asset_count=10,
        start_date=Config.DEFAULT_START_DATE,
        end_date=Config.DEFAULT_END_DATE,
        mu=None,
        tr=None,
        df_perf=None,
        df_alloc=None,
        corr_matrix=None,
        target_return=Config.TARGET_RETURN,
        target_volatility=Config.TARGET_RISK,
        error=None
    )

if __name__ == "__main__":
    app.run(debug=True)