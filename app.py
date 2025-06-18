from flask import Flask, request, jsonify, render_template, send_file
from data_service import ParquetDataService
from cache_utils import make_cache_key, save_cache, load_cache, cache_exists
from optimizing import PortfolioOptimizer
from charting import Chart
import pandas as pd
from io import BytesIO
import os
import re
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
    RISK_FREE_RATE = 0.04
    TARGET_RETURN = 0.23
    TARGET_RISK = 0.22

@app.route('/api/tickers')
def tickers_api():
    q = request.args.get('q', '').upper()
    all_tickers = data_service.get_tickers
    matches = [t for t in all_tickers if t.startswith(q)]
    return jsonify(matches[:20])

@app.route('/export_weights', methods=['POST'])
def export_weights():
    try:
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        rebalance_freq = request.form.get('rebalance')
        selected_tickers = request.form.get("selected_tickers", "").split(",")
        # Add these missing values from the form:
        target_return = float(request.form.get('target_return', 0.05))
        target_volatility = float(request.form.get('target_volatility', 0.10))
        # Reconstruct min_weights and max_weights from form
        min_weights = {}
        max_weights = {}
        for i, ticker in enumerate(selected_tickers):
            min_val = float(request.form.get(f"min_{i}", 0.0)) / 100
            max_val = float(request.form.get(f"max_{i}", 15.0)) / 100
            min_weights[ticker] = min_val
            max_weights[ticker] = max_val

        # Now cache_key will work correctly
        cache_key = make_cache_key(
            selected_tickers, start_date, end_date,
            target_return, target_volatility,
            min_weights, max_weights
        )

        cached_data = load_cache(cache_key)
        strategies = cached_data['alloc_dict']
        weights_by_strategy = data_service.compute_weights_for_strategies(strategies, start_date, end_date, rebalance=rebalance_freq)
        if not weights_by_strategy:
            return "No weight data available for this rebalance frequency.", 400

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for strategy_name, weights_df in weights_by_strategy.items():
                clean = re.sub(r'[:\\/?*\[\]]', '', strategy_name)
                weights_df.to_excel(writer, sheet_name=clean[:31], index=False)

        output.seek(0)
        filename = f"weights_{rebalance_freq}_{start_date}_to_{end_date}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return f"Error exporting weights: {e}", 500
    
@app.route('/plot/efficient_frontier')
def plot_efficient_frontier():
    cache_key = request.args.get("cache_key")
    if not cache_key or not cache_exists(cache_key):
        return jsonify({'html': "<p>Cache not found.</p>"}), 400

    cached_data = load_cache(cache_key)
    fig_html = cached_data.get('efficient_frontier_figure')
    if not fig_html:
        return jsonify({'html': "<p>Efficient Frontier plot not cached.</p>"}), 400

    return jsonify({'html': fig_html})

@app.route('/plot/nav_chart')
def plot_nav_chart():
    cache_key = request.args.get("cache_key")
    rebalance = request.args.get("rebalance", "monthly")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not all([cache_key, start_date, end_date]):
        return jsonify({'html': "<p>Missing required parameters.</p>"}), 400

    if not cache_exists(cache_key):
        return jsonify({'html': "<p>Cache not found.</p>"}), 400

    cached_data = load_cache(cache_key)
    portfolio_plots = cached_data.get('portfolio_plots', {})
    plot_html = portfolio_plots.get(rebalance)

    if not plot_html:
        return jsonify({'html': "<p>Plot not cached for this rebalance.</p>"}), 400

    return jsonify({'html': plot_html})

@app.route('/plot/heatmap')
def plot_heatmap():
    cache_key = request.args.get("cache_key")

    if not cache_key or not cache_exists(cache_key):
        return jsonify({'html': "<p>Heatmap data not found.</p>"}), 400

    cached_data = load_cache(cache_key)
    alloc_dict = cached_data.get('alloc_dict')

    if not alloc_dict:
        return jsonify({'html': "<p>Allocation data missing.</p>"}), 400

    strategy_weights = next(iter(alloc_dict.values()), {})
    tickers = list(strategy_weights.keys())

    chart = Chart(data_service)
    fig = chart.heatmap(tickers)
    return jsonify({'html': fig})


@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute") 
def unified_portfolio():
    context = {
        'selected_tickers': Config.SAMPLE_TICKERS,
        'asset_count': 10,
        'start_date': Config.DEFAULT_START_DATE,
        'end_date': Config.DEFAULT_END_DATE,
        'mu': None,
        'vol': None,
        'sharpe': None,
        'perf_dict': None,
        'alloc_dict': None,
        'corr_matrix': None,
        'min_weights': {},
        'max_weights': {},
        'target_return': Config.TARGET_RETURN,
        'target_volatility': Config.TARGET_RISK,
        'cache_key': None,
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
                    min_weights[ticker] = float(request.form.get(f"min_{i}", 0.0)) / 100
                    max_weights[ticker] = float(request.form.get(f"max_{i}", 20.0)) / 100

            if target_return < 0 or target_volatility < 0:
                context['error'] = "Target return and volatility must be positive."
                return render_template('unified_portfolio.html', **context)

            if any(min_weights[t] > max_weights[t] for t in selected_tickers if t in min_weights):
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

            cache_key = make_cache_key(
                selected_tickers, start_date, end_date,
                target_return, target_volatility,
                min_weights, max_weights
            )

            if cache_exists(cache_key):
                cached_data = load_cache(cache_key)
                context.update({
                    'perf_dict': cached_data['perf_dict'],
                    'alloc_dict': cached_data['alloc_dict'],
                    'corr_matrix': cached_data['corr_matrix'],
                    'mu': cached_data.get('mu'),
                    'vol': cached_data.get('vol'),
                    'sharpe': cached_data.get('sharpe'),
                    'cache_key': cache_key  # Important for JS lazy-loading
                })
                return render_template('unified_portfolio.html', **context)

            # Compute everything from scratch
            returns_df = data_service.get_returns(selected_tickers, start_date, end_date)
            missing_returns = [t for t in selected_tickers if t not in returns_df.index or pd.isna(returns_df.loc[t, 'Annualized Return'])]

            if missing_returns:
                context['error'] = f"No return data for tickers: {', '.join(missing_returns)}"
                return render_template('unified_portfolio.html', **context)

            mu = returns_df.loc[selected_tickers, 'Annualized Return']
            prices_df = data_service.get_prices(selected_tickers, start_date, end_date)
            missing_prices = [t for t in selected_tickers if t not in prices_df.columns or prices_df[t].dropna().empty]
            if missing_prices:
                context['error'] = f"No price data for tickers: {', '.join(missing_prices)}"
                return render_template('unified_portfolio.html', **context)

            tickers_available = prices_df.columns.intersection(mu.index)
            mu = mu.loc[tickers_available]
            prices_df = prices_df[tickers_available]

            bounds = [(min_weights.get(t, 0.0), max_weights.get(t, 0.2)) for t in mu.index]
            optimizer = PortfolioOptimizer()
            results = optimizer.run_optimizations(
                mu, prices_df,
                bounds=bounds,
                target_return=target_return,
                target_volatility=target_volatility
            )
            ef_max_sharpe = results["Max Sharpe"]
            daily_returns = prices_df.pct_change().dropna()
            vol = daily_returns.std() * (252 ** 0.5)
            sharpe = (mu - Config.RISK_FREE_RATE) / vol

            
            corr_matrix = PortfolioOptimizer.compute_corr_matrix(prices_df)

            perf_dict, alloc_dict = optimizer.compile_results(results, Config.RISK_FREE_RATE)

            chart = Chart(data_service)
            ef_frontier = chart.plot_efficient_frontier(ef_max_sharpe, mu,vol)
            heatmap_plot = chart.heatmap(tickers_available.tolist())

            portfolio_plots = {}
            for freq in ['weekly', 'monthly']:
                navs = data_service.compute_navs_for_strategies(alloc_dict, start_date, end_date, rebalance=freq)
                portfolio_plots[freq] = chart.plot_portfolios(navs, rebalance=freq)
            
            # Save to cache
            save_cache(cache_key, {
                'mu': mu,
                'vol': vol,
                'sharpe': sharpe,
                'perf_dict': perf_dict,
                'alloc_dict': alloc_dict,
                'corr_matrix': corr_matrix,
                'efficient_frontier_figure': ef_frontier,
                'heatmap_figure': heatmap_plot,
                'portfolio_plots': portfolio_plots,
            })
            context.update({
                'mu': mu,
                'vol': vol,
                'sharpe': sharpe,
                'perf_dict': perf_dict,
                'alloc_dict': alloc_dict,
                'corr_matrix': corr_matrix,
                'cache_key': cache_key,
            })

        except Exception as e:
            context['error'] = f"Internal Server Error: {e}"

    return render_template('unified_portfolio.html', **context)

@app.route('/health')
def health():
    return 'OK', 200

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8000)
    app.run(debug=True)
