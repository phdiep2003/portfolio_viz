from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json, pickle
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "123459876"  # TODO: Replace with secure secret key in production

prices_df = pd.read_parquet('data/prices.parquet')
ticker_sector_df = pd.read_csv('tickers_with_sectors.csv')

# Assuming the ticker column is named 'ticker' or similar
unique_tickers = sorted(prices_df['Ticker'].unique())

# --- Configuration ---
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

class Config:
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'MS', 'T', 'XOM','NEM']
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2025-01-01"
    RISK_FREE_RATE = 0.03
    TARGET_RETURN = 0.23
    TARGET_RISK = 0.22
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
            f"Efficient Return (target: {target_return:.3f})": lambda ef: ef.efficient_return(target_return),
            f"Efficient Risk (target: {target_volatility:.3f})": lambda ef: ef.efficient_risk(target_volatility),
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

class Chart:
    @staticmethod
    def create_efficient_frontier_plot(mu, vol, sharpe, ef_returns, ef_volatility):
        """
        Create interactive efficient frontier plot with Plotly Express
        
        Parameters:
        - mu: Expected returns of individual assets
        - vol: Volatility of individual assets
        - sharpe: Sharpe ratios of individual assets
        - ef_returns: Efficient frontier portfolio returns
        - ef_volatility: Efficient frontier portfolio volatilities
        """
        # Create DataFrame for individual assets
        assets_df = pd.DataFrame({
            'Ticker': mu.index,
            'Expected Return': mu * 100,
            'Volatility': vol * 100,
            'Sharpe Ratio': sharpe
        })
        
        # Sort frontier points by volatility to ensure proper line drawing
        frontier_points = sorted(zip(ef_volatility * 100, ef_returns * 100), key=lambda x: x[0])
        frontier_vol, frontier_ret = zip(*frontier_points)
        
        # Create base figure with individual assets
        fig = go.Figure()
        
        # Add individual assets with color based on Sharpe ratio (but without colorbar)
        for _, row in assets_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row['Volatility']],
                    y=[row['Expected Return']],
                    mode='markers+text',
                    name=row['Ticker'],
                    marker=dict(
                        size=12,
                        color=row['Sharpe Ratio'],
                        colorscale='Viridis',
                        showscale=False,  # This removes the colorbar
                        line=dict(width=1, color='black')
                    ),
                    text=[row['Ticker']],
                    textposition='top center',
                    hovertemplate=
                        '<b>'+row['Ticker']+'</b><br>'+
                        'Volatility: %{x:.2f}%<br>' +
                        'Return: %{y:.2f}%<br>' +
                        'Sharpe: %.2f<extra></extra>' % row['Sharpe Ratio'],
                    showlegend=False
                )
            )
        
        # Add efficient frontier line (now properly sorted)
        fig.add_trace(
            go.Scatter(
                x=frontier_vol,
                y=frontier_ret,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='red', width=3),
                hovertemplate='<b>Volatility</b>: %{x:.2f}%<br><b>Return</b>: %{y:.2f}%'
            )
        )
        
        # Highlight max Sharpe ratio portfolio
        max_sharpe_idx = np.argmax((ef_returns - 0.02) / ef_volatility)  # Assuming 2% risk-free rate
        fig.add_trace(
            go.Scatter(
                x=[ef_volatility[max_sharpe_idx] * 100],
                y=[ef_returns[max_sharpe_idx] * 100],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='green',
                    line=dict(width=1, color='black')
                ),
                hovertemplate=
                    '<b>Max Sharpe Portfolio</b><br>' +
                    'Volatility: %{x:.2f}%<br>' +
                    'Return: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Add minimum volatility portfolio (leftmost point)
        min_vol_idx = np.argmin(ef_volatility)
        fig.add_trace(
            go.Scatter(
                x=[ef_volatility[min_vol_idx] * 100],
                y=[ef_returns[min_vol_idx] * 100],
                mode='markers',
                name='Min Volatility',
                marker=dict(
                    symbol='diamond',
                    size=20,
                    color='orange',
                    line=dict(width=1, color='black')
                ),
                hovertemplate=
                    '<b>Min Volatility Portfolio</b><br>' +
                    'Volatility: %{x:.2f}%<br>' +
                    'Return: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Annualized Return (%)',
            hovermode='closest',
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    @staticmethod
    def heatmap(selected_tickers):
        filtered_df = ticker_sector_df[ticker_sector_df['Ticker'].isin(selected_tickers)].copy()
        
        # Create equal weights for uniform box sizes
        filtered_df['Weight'] = 1
        
        # Create color mapping for sectors only
        unique_sectors = filtered_df['Sector'].unique()
        sector_colors = px.colors.qualitative.G10[:len(unique_sectors)]
        color_map = dict(zip(unique_sectors, sector_colors))
        
        # Create the treemap with all boxes initially white
        fig = px.treemap(
            filtered_df,
            path=['Sector', 'Ticker'],
            values='Weight',
            color_discrete_sequence=['white']  # Start with all white
        )
        
        # Custom coloring - only color the sector (parent) boxes
        colors = []
        text_colors = []
        for label, parent in zip(fig.data[0].labels, fig.data[0].parents):
            if parent == '':  # This is a sector box
                colors.append(color_map[label])
                text_colors.append('white')  # White text for colored sectors
            else:  # This is a stock box
                colors.append('white')
                text_colors.append('black')  # Black text for white stocks
        
        # Apply custom styling
        fig.update_traces(
            marker_colors=colors,
            textfont_color=text_colors,
            textinfo="label",
            textposition="middle center",
            textfont=dict(size=16),
            marker=dict(
                line=dict(width=1, color="darkgray")
            ),
            hovertemplate='<b>%{label}</b><br>Sector: %{parent}<extra></extra>',
            branchvalues='total',
            tiling=dict(
                packing='squarify',
                squarifyratio=1  # More uniform box sizes
            )
        )
        
        fig.update_layout(
            margin=dict(t=30, l=0, r=0, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            uniformtext=dict(
                minsize=12,
                mode='hide'
            ),
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn') 
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
        return merged[['Ticker','Years','Total Return', 'Annualized Return']]

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
    context = {
        'selected_tickers': Config.SAMPLE_TICKERS,
        'asset_count': 10,
        'start_date': Config.DEFAULT_START_DATE,
        'end_date': Config.DEFAULT_END_DATE,
        'mu': None,
        'efficient_frontier_plot,': None,
        'heatmap_plot': None,
        'vol': None,
        'sharpe': None,
        'df_perf': None,
        'df_alloc': None,
        'corr_matrix': None,
        'min_weights': {},
        'max_weights': {},
        'target_return': Config.TARGET_RETURN,
        'target_volatility': Config.TARGET_RISK,
        'error': None
    }

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

            context.update({
                'selected_tickers': selected_tickers,
                'asset_count': len(selected_tickers),
                'start_date': start_date,
                'end_date': end_date,
                'min_weights': min_weights,
                'max_weights': max_weights,
                'target_return': target_return,
                'target_volatility': target_volatility
            })

            # Generate cache key
            cache_key_val = make_cache_key(
                selected_tickers, start_date, end_date,
                target_return, target_volatility,
                min_weights, max_weights
            )

            # Check if cached result exists
            if cache_exists(cache_key_val):
                cached_data = load_cache(cache_key_val)
                context.update({
                    'mu': cached_data['mu'],
                    'vol': cached_data['vol'],
                    'sharpe': cached_data['sharpe'],
                    'df_perf': cached_data['df_perf'],
                    'df_alloc': cached_data['df_alloc'],
                    'corr_matrix': cached_data['corr_matrix'],
                    'efficient_frontier_plot': cached_data['efficient_frontier_plot'],
                    'heatmap_plot': cached_data['heatmap_plot']
                })
            else:
                
                # === No cache found, do full calculation ===
                data_service = ParquetDataService()
                returns_df = data_service.get_returns(selected_tickers, start_date, end_date)

                # Check for missing return data per ticker
                missing_returns = [t for t in selected_tickers if t and returns_df[returns_df['Ticker'] == t].dropna().empty]
                if missing_returns:
                    context['error'] = f"No return data for tickers: {', '.join(missing_returns)}"
                else:
                    mu = returns_df.groupby('Ticker')['Annualized Return'].first()
                    # tr = returns_df.groupby('Ticker')['Total Return'].first()
                    # yrs = returns_df.groupby('Ticker')['Years'].first()                    
                    prices_df = data_service.get_prices(selected_tickers, start_date, end_date)
                    # Check for missing price data per ticker
                    missing_prices = [t for t in selected_tickers if t and prices_df[prices_df['Ticker'] == t].empty]
                    if missing_prices:
                        context['error'] = f"No price data for tickers: {', '.join(missing_prices)}"
                    else:
                        print("hello")
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
                        vol = per_asset_metrics.set_index('Ticker')['Volatility']
                        sharpe = per_asset_metrics.set_index('Ticker')['Sharpe Ratio']
                        df_perf, df_alloc = optimizer.compile_results(results)
                        corr_matrix = PortfolioOptimizer.compute_corr_matrix(price_pivot, '_'.join(sorted(price_pivot.columns)))
                        # Create efficient frontier plot
                        ef_returns = np.array([result.expected_return for result in results.values() if isinstance(result, OptimizationResult)])
                        ef_volatility = np.array([result.volatility for result in results.values() if isinstance(result, OptimizationResult)])

                        efficient_frontier_plot = Chart.create_efficient_frontier_plot(
                            mu, vol, sharpe,
                            ef_returns, ef_volatility
                        )
                        heatmap_plot = Chart.heatmap(selected_tickers)
                        # Save cache
                        print("HI")
                        save_cache(cache_key_val, {
                            'mu': mu,
                            'vol': vol,
                            'sharpe': sharpe,
                            'df_perf': df_perf,
                            'df_alloc': df_alloc,
                            'corr_matrix': corr_matrix,
                            'efficient_frontier_plot': efficient_frontier_plot,
                            'heatmap_plot': heatmap_plot
                        })
                        context.update({
                            'mu': mu,
                            'vol': vol,
                            'sharpe': sharpe,
                            'df_perf': df_perf,
                            'df_alloc': df_alloc,
                            'corr_matrix': corr_matrix,
                            'efficient_frontier_plot': efficient_frontier_plot,
                            'heatmap_plot': heatmap_plot
                        })

        except Exception as e:
            context['error'] = f"Internal Server Error: {e}"
    return render_template('unified_portfolio.html', **context)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)