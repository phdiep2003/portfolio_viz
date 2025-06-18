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
    return jsonify(matches[:20])  # limit to 20 results

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
        cache_key_val = make_cache_key(
            selected_tickers, start_date, end_date,
            target_return, target_volatility,
            min_weights, max_weights
        )

        cached_data = load_cache(cache_key_val)
        strategies = cached_data['df_alloc']
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
        'df_perf': None,
        'df_alloc': None,
        'corr_matrix': None,
        'min_weights': {},
        'max_weights': {},
        'target_return': Config.TARGET_RETURN,
        'target_volatility': Config.TARGET_RISK,
        'efficient_frontier_plot': None, 
        'heatmap_plot': None,
        'portfolio_plots': None,
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
            # print(selected_tickers, start_date, end_date,
            #     target_return, target_volatility,
            #     min_weights, max_weights)
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
                    'heatmap_plot': cached_data['heatmap_plot'],
                    'portfolio_plots': cached_data['portfolio_plots'],
                })
            else:
                returns_df = data_service.get_returns(selected_tickers, start_date, end_date)
                # Find tickers missing return data or with empty Annualized Return
                missing_returns = [t for t in selected_tickers if t not in returns_df.index or pd.isna(returns_df.loc[t, 'Annualized Return'])]

                if missing_returns:
                    context['error'] = f"No return data for tickers: {', '.join(missing_returns)}"
                else:
                    # Select mu as Series of Annualized Returns indexed by ticker
                    mu = returns_df.loc[selected_tickers, 'Annualized Return']

                    # Get price data
                    prices_df = data_service.get_prices(selected_tickers, start_date, end_date)

                    # Check for missing price data (column missing or all NaN)
                    missing_prices = [t for t in selected_tickers if t not in prices_df.columns or prices_df[t].dropna().empty]
                    if missing_prices:
                        context['error'] = f"No price data for tickers: {', '.join(missing_prices)}"
                    else:
                        # Filter prices by date range
                        price_pivot = data_service.filter_date_range(prices_df, start_date, end_date)

                        # Align mu and price_pivot columns
                        tickers_available = price_pivot.columns.intersection(mu.index)
                        mu = mu.loc[tickers_available]
                        price_pivot = price_pivot[tickers_available]

                        optimizer = PortfolioOptimizer()
                        bounds = [(min_weights.get(t, 0.0), max_weights.get(t, 0.2)) for t in mu.index]

                        results, newplot = optimizer.run_optimizations(
                            mu, price_pivot,
                            bounds=bounds,
                            target_return=target_return,
                            target_volatility=target_volatility
                        )

                        daily_returns = price_pivot.pct_change().dropna()
                        vol = daily_returns.std() * (252 ** 0.5)
                        sharpe = (mu - Config.RISK_FREE_RATE) / vol
                        corr_matrix = PortfolioOptimizer.compute_corr_matrix(price_pivot)

                        # print(results)
                        df_perf, df_alloc = optimizer.compile_results(results, Config.RISK_FREE_RATE)

                        ## Plotting ##
                        chart = Chart(data_service)
                        heatmap_plot = chart.heatmap(tickers_available.tolist())

                        strategies = df_alloc.set_index('Strategy').T.to_dict()
                        rebalance_options = ['weekly', 'monthly']
                        portfolio_plots = {}
                        for freq in rebalance_options:
                            navs = data_service.compute_navs_for_strategies(strategies, start_date, end_date, rebalance=freq)
                            portfolio_plots[freq] = chart.plot_portfolios(navs, rebalance=freq)

                        # Serialize plots to JSON for caching
                        save_cache(cache_key_val, {
                            'mu': mu,
                            'vol': vol,
                            'sharpe': sharpe,
                            'df_perf': df_perf,
                            'df_alloc': strategies,
                            'corr_matrix': corr_matrix,
                            'efficient_frontier_plot': newplot,
                            'heatmap_plot': heatmap_plot,
                            'portfolio_plots': portfolio_plots,
                        })

                        context.update({
                            'mu': mu,
                            'vol': vol,
                            'sharpe': sharpe,
                            'df_perf': df_perf,
                            'df_alloc': strategies,
                            'corr_matrix': corr_matrix,
                            'efficient_frontier_plot': newplot,
                            'heatmap_plot': heatmap_plot,
                            'portfolio_plots': portfolio_plots,
                        })

        except Exception as e:
            context['error'] = f"Internal Server Error: {e}"
    return render_template('unified_portfolio.html', **context)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
