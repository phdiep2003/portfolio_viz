from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import orjson
# Local lightweight services
from flask_compress import Compress
from data_service import ParquetDataService
from cache_utils import FileCache

app = Flask(__name__)
Compress(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")
app.env = os.getenv("FLASK_ENV", "production")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

data_service = ParquetDataService()
cache = FileCache()

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)

class Config:
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'MS', 'T', 'XOM', 'NEM']
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2025-01-01"
    RISK_FREE_RATE = 0.04
    TARGET_RETURN = 0.23
    TARGET_RISK = 0.22

@app.route('/api/tickers')
def tickers_api():
    q = request.args.get('q', '').upper()
    matches = [t for t in data_service.get_tickers if t.startswith(q)]
    return Response(orjson.dumps(matches[:20]), content_type="application/json")

@app.route('/plot/<string:plot_type>')
def plot_handler(plot_type):
    cache_key = request.args.get("cache_key")
    if not cache_key or not cache.exists(cache_key):
        return jsonify({'error': f"{plot_type.title()} data not found."}), 400

    cached_bytes = cache.load(cache_key)  # load bytes from cache
    cached = orjson.loads(cached_bytes)   # parse bytes to dict

    if plot_type == "efficient_frontier":
        data = cached.get("efficient_frontier_data")
        layout = cached.get("efficient_frontier_layout")
    elif plot_type == "heatmap":
        data = cached.get("heatmap_data")
        layout = cached.get("heatmap_layout")
    elif plot_type == "nav_chart":
        freq = request.args.get("rebalance", "monthly")
        data_all = cached.get("portfolio_data", {})
        layout_all = cached.get("portfolio_layout", {})
        data = data_all.get(freq)
        layout = layout_all.get(freq)
    else:
        return jsonify({'error': "Invalid plot type."}), 400

    if not data or not layout:
        return jsonify({'error': f"{plot_type.title()} figure missing."}), 400

    if not isinstance(data, list):
        data = [data]

    return Response(
        orjson.dumps({'data': data, 'layout': layout}),
        content_type="application/json"
    )

@app.route('/export_weights', methods=['POST'])
def export_weights():
    import pandas as pd
    import re
    from io import BytesIO
    try:
        selected = request.form.get("selected_tickers", "").split(",")
        start, end = request.form.get('start_date'), request.form.get('end_date')
        r_target = float(request.form.get('target_return', 0.05))
        v_target = float(request.form.get('target_volatility', 0.10))
        rebalance = request.form.get('rebalance')

        min_w = {t: float(request.form.get(f"min_{i}", 0)) / 100 for i, t in enumerate(selected)}
        max_w = {t: float(request.form.get(f"max_{i}", 15)) / 100 for i, t in enumerate(selected)}
        
        cache_key = cache.make_cache_key(selected, start, end, r_target, v_target, min_w, max_w)
        cached_bytes = cache.load(cache_key)
        cached = orjson.loads(cached_bytes)
        alloc_dict = cached.get('alloc_dict')
        weights = data_service.compute_weights_for_strategies(alloc_dict, start, end, rebalance=rebalance)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for strat, df in weights.items():
                df.to_excel(writer, sheet_name=re.sub(r'[:\\/?*\[\]]', '', strat)[:31], index=False)

        output.seek(0)
        filename = f"weights_{rebalance}_{start}_to_{end}.xlsx"
        return send_file(output, as_attachment=True, download_name=filename)
    except Exception as e:
        return f"Export failed: {e}", 500

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def unified_portfolio():
    global np, PortfolioOptimizer, Chart, find_non_serializable
    if 'np' not in globals():
        import numpy as np
    if 'PortfolioOptimizer' not in globals():
        from optimizing import PortfolioOptimizer
    if 'Chart' not in globals():
        from charting import Chart
    if 'find_non_serializable' not in globals():
        from serialization import find_non_serializable
    context = {
        'selected_tickers': Config.SAMPLE_TICKERS,
        'asset_count': 10,
        'start_date': Config.DEFAULT_START_DATE,
        'end_date': Config.DEFAULT_END_DATE,
        'mu': None, 'vol': None, 'sharpe': None,
        'perf_dict': None, 'alloc_dict': None, 'corr_matrix': None,
        'min_weights': {}, 'max_weights': {},
        'target_return': Config.TARGET_RETURN,
        'target_volatility': Config.TARGET_RISK,
        'cache_key': None, 'error': None
    }
    def serialize_fig(fig_data, fig_layout):
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj

        data_json = [convert(trace) for trace in fig_data]
        layout_json = convert(fig_layout)
        return data_json, layout_json

    if request.method == 'POST':
        try:
            count = int(request.form.get('asset_count', 10))
            tickers = [request.form.get(f'tickers_{i}') for i in range(count) if request.form.get(f'tickers_{i}')]
            start, end = request.form.get('start_date'), request.form.get('end_date')
            r_target = float(request.form.get('target_return', Config.TARGET_RETURN))
            v_target = float(request.form.get('target_volatility', Config.TARGET_RISK))

            min_w = {t: float(request.form.get(f"min_{i}", 0)) / 100 for i, t in enumerate(tickers)}
            max_w = {t: float(request.form.get(f"max_{i}", 20)) / 100 for i, t in enumerate(tickers)}

            if any(min_w[t] > max_w[t] for t in tickers):
                context['error'] = "Min weight can't exceed max weight."
                context.update(selected_tickers=tickers, asset_count=len(tickers), start_date=start, end_date=end,
                               min_weights=min_w, max_weights=max_w, target_return=r_target, target_volatility=v_target)
                return render_template('unified_portfolio.html', **context)

            context.update(selected_tickers=tickers, asset_count=len(tickers), start_date=start, end_date=end,
                           min_weights=min_w, max_weights=max_w, target_return=r_target, target_volatility=v_target)

            cache_key = cache.make_cache_key(tickers, start, end, r_target, v_target, min_w, max_w)
            context['cache_key'] = cache_key

            cached_bytes = cache.load(cache_key)
            if cached_bytes is not None:
                cached_data = orjson.loads(cached_bytes)
                context.update(cached_data)
                return render_template('unified_portfolio.html', **context)

            # Cache miss: compute all fresh
            returns_df = data_service.get_returns(tickers, start, end)
            if returns_df.empty:
                context['error'] = "No return data found."
                return render_template('unified_portfolio.html', **context)

            mu = returns_df.loc[tickers, 'Annualized Return']
            prices = data_service.get_prices(tickers, start, end)
            common_cols = [col for col in prices.columns if col in mu.index]
            prices = prices[common_cols]

            bounds = [(min_w[t], max_w[t]) for t in mu.index]

            optimizer = PortfolioOptimizer()
            results = optimizer.run_optimizations(mu, prices, bounds, r_target, v_target)
            ef = results['Max Sharpe']

            vol = prices.pct_change().dropna().std() * (252 ** 0.5)
            sharpe = (mu - Config.RISK_FREE_RATE) / vol
            corr = PortfolioOptimizer.compute_corr_matrix(prices)
            perf, alloc = optimizer.compile_results(results, Config.RISK_FREE_RATE)
            chart = Chart(data_service)

            # Efficient Frontier plot
            data_ef, layout_ef = chart.plot_efficient_frontier(ef, mu, vol)
            data_ef_json, layout_ef_json = serialize_fig(data_ef, layout_ef)

            # Heatmap plot
            data_heatmap, layout_heatmap = chart.heatmap(mu.index.tolist())
            data_heatmap_json, layout_heatmap_json = serialize_fig(data_heatmap, layout_heatmap)

            # Portfolio plots for monthly and weekly
            port_data = {}
            port_layout = {}
            # for freq in ['monthly', 'weekly']:
            navs = data_service.compute_navs_for_strategies(alloc, start, end, 'monthly')
            data_port, layout_port = chart.plot_portfolios(navs, 'monthly')
            data_port_json, layout_port_json = serialize_fig(data_port, layout_port)
            port_data['monthly'] = data_port_json
            port_layout['monthly'] = layout_port_json
            result_data = {
                'mu': mu.to_dict(),
                'vol': vol.to_dict(),
                'sharpe': sharpe.to_dict(),
                'perf_dict': perf,
                'alloc_dict': alloc,
                'corr_matrix': corr,
                'efficient_frontier_data': data_ef_json,
                'efficient_frontier_layout': layout_ef_json,
                'heatmap_data': data_heatmap_json,
                'heatmap_layout': layout_heatmap_json,
                'portfolio_data': port_data,
                'portfolio_layout': port_layout
            }
            problem_key = find_non_serializable(result_data)
            if problem_key:
                print(f"Non-serializable data found at: {problem_key}")
            else:
                print("All data is JSON serializable.")
            result_bytes = orjson.dumps(result_data)

            # Save bytes to cache
            cache.save(cache_key, result_bytes)

            # Update context from dict
            context.update(result_data)
        except Exception as e:
            context['error'] = f"Error: {e}"

    return render_template('unified_portfolio.html', **context)

@app.route('/health')
def health():
    return 'OK', 200