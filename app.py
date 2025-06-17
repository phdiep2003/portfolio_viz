from flask import Flask, request, jsonify, render_template
from data_service import ParquetDataService
from cache_utils import make_cache_key, save_cache, load_cache, cache_exists
from optimizing import PortfolioOptimizer
from charting import Chart
import os
import warnings
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")  # Fallback for local dev
app.env = os.getenv("FLASK_ENV", "development")  # Default to development if not set
data_service = ParquetDataService() 

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Limits by IP
    storage_uri="memory://",  # Default in-memory storage
    default_limits=["200 per day", "50 per hour"]  # Global limits
)

class Config:
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'MS', 'T', 'XOM','NEM']
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2025-01-01"
    RISK_FREE_RATE = 0.03
    TARGET_RETURN = 0.23
    TARGET_RISK = 0.22

@app.route('/api/tickers')
def tickers_api():
    q = request.args.get('q', '').upper()
    all_tickers = data_service.get_tickers()
    matches = [t for t in all_tickers if t.startswith(q)]
    return jsonify(matches[:20])  # limit to 20 results

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def unified_portfolio():
    context = {
        'selected_tickers': Config.SAMPLE_TICKERS,
        'asset_count': 10,
        'start_date': Config.DEFAULT_START_DATE,
        'end_date': Config.DEFAULT_END_DATE,
        'mu': None,
        'efficient_frontier_plot': None,  # fixed comma typo here
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
                if ticker:
                    min_value = float(request.form.get(f"min_{i}", 0.0)) / 100
                    max_value = float(request.form.get(f"max_{i}", 20.0)) / 100
                    min_weights[ticker] = min_value
                    max_weights[ticker] = max_value
            if target_return < 0 or target_volatility < 0:
                context['error'] = "Target return and volatility must be positive."
                return render_template('unified_portfolio.html', **context)

            if any(min_weights[t] > max_weights[t] for t in selected_tickers):
                context['error'] = "Min weight cannot be greater than max weight for any asset."
                return render_template('unified_portfolio.html', **context)

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

            cache_key_val = make_cache_key(
                selected_tickers, start_date, end_date,
                target_return, target_volatility,
                min_weights, max_weights
            )

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
                returns_df = data_service.get_returns(selected_tickers, start_date, end_date)
                missing_returns = [t for t in selected_tickers if t and returns_df[returns_df['Ticker'] == t].dropna().empty]
                if missing_returns:
                    context['error'] = f"No return data for tickers: {', '.join(missing_returns)}"
                else:
                    mu = returns_df.groupby('Ticker')['Annualized Return'].first()
                    prices_df = data_service.get_prices(selected_tickers, start_date, end_date)

                    missing_prices = [t for t in selected_tickers if t and prices_df[prices_df['Ticker'] == t].empty]
                    if missing_prices:
                        context['error'] = f"No price data for tickers: {', '.join(missing_prices)}"
                    else:
                        prices_df = data_service.filter_date_range(prices_df, 'Date', start_date, end_date)
                        price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
                        mu = mu.loc[price_pivot.columns]

                        optimizer = PortfolioOptimizer()
                        bounds = [(min_weights.get(t, 0.0), max_weights.get(t, 0.2)) for t in mu.index]
                        results = optimizer.run_optimizations(
                            mu, price_pivot,
                            bounds=bounds,
                            risk_free_rate=Config.RISK_FREE_RATE,
                            target_return=target_return,
                            target_volatility=target_volatility
                        )

                        daily_returns = price_pivot.pct_change().dropna()
                        vol = daily_returns.std() * (252 ** 0.5)
                        sharpe = (mu - Config.RISK_FREE_RATE) / vol
                        corr_matrix = PortfolioOptimizer.compute_corr_matrix(price_pivot)
                        df_perf, df_alloc = optimizer.compile_results(results, Config.RISK_FREE_RATE)

                        ef_returns = df_perf["Expected Return"].values
                        ef_volatility = df_perf["Volatility"].values

                        efficient_frontier_plot = Chart.create_efficient_frontier_plot(
                            mu, vol, sharpe, ef_returns, ef_volatility
                        )
                        heatmap_plot = Chart.heatmap(selected_tickers)
                        # Serialize plots to JSON for caching
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
    # app.run(host='0.0.0.0', port=8000)
    app.run(debug=True)