from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import orjson
from htmlmin.main import minify
from flask_compress import Compress
import numpy as np
import pandas as pd
import re
from io import BytesIO

# Local lightweight services
from data_service import ParquetDataService
from cache_utils import FileCache
from optimizing import PortfolioOptimizer
from charting import Chart

app = Flask(__name__)
Compress(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")

# --- Initialize Services ---
data_service = ParquetDataService()
cache = FileCache()
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)

# --- Configuration ---
class Config:
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'MS', 'T', 'XOM', 'NEM']
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2025-01-01"
    RISK_FREE_RATE = 0.04
    TARGET_RETURN = 0.23
    TARGET_RISK = 0.22

# --- Preload Default Portfolio Results (baked JSON) ---
DEFAULT_RESULT = None
default_path = os.path.join(os.path.dirname(__file__), "data", "default_portfolio.json")
if os.path.exists(default_path):
    with open(default_path, "rb") as f:
        DEFAULT_RESULT = orjson.loads(f.read())
        
# --- ROUTE 1: Main Page (GET only) ---
@app.route('/', methods=['GET'])
def unified_portfolio():
    """
    Renders the main portfolio visualizer page.
    This route ONLY handles the initial page load.
    """
    context = {
        'selected_tickers': Config.SAMPLE_TICKERS,
        'asset_count': len(Config.SAMPLE_TICKERS),
        'start_date': Config.DEFAULT_START_DATE,
        'end_date': Config.DEFAULT_END_DATE,
        'target_return': Config.TARGET_RETURN,
        'target_volatility': Config.TARGET_RISK,
    }
    rendered = render_template("unified_portfolio.html", **context)
    return minify(rendered, remove_empty_space=True)

# --- ROUTE 2: API for Portfolio Calculations (POST only) ---
@app.route('/api/portfolio_analysis', methods=['POST'])
@limiter.limit("10 per minute")
def portfolio_analysis_api():
    """
    Receives portfolio parameters via JSON, performs calculations,
    and returns all chart and table data as JSON.
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        start = data.get('start_date', Config.DEFAULT_START_DATE)
        end = data.get('end_date', Config.DEFAULT_END_DATE)

        # --- FIX: Safely handle potentially missing values ---
        raw_r_target = data.get('target_return')
        r_target = float(raw_r_target) if raw_r_target else Config.TARGET_RETURN

        raw_v_target = data.get('target_volatility')
        v_target = float(raw_v_target) if raw_v_target else Config.TARGET_RISK
        # --- END FIX ---
        min_w = {t: float(w) / 100 for t, w in data.get('min_weights', {}).items()}
        max_w = {t: float(w) / 100 for t, w in data.get('max_weights', {}).items()}
        if (
            tickers == Config.SAMPLE_TICKERS
            and start == Config.DEFAULT_START_DATE
            and end == Config.DEFAULT_END_DATE
            and abs(r_target - Config.TARGET_RETURN) < 1e-9
            and abs(v_target - Config.TARGET_RISK) < 1e-9
            and min_w == {t: 0 for t in Config.SAMPLE_TICKERS}
            and max_w == {t: 0.15 for t in Config.SAMPLE_TICKERS}
        ):
            if DEFAULT_RESULT is not None:
                return Response(
                    orjson.dumps(DEFAULT_RESULT),
                    content_type="application/json"
                )
        if not tickers:
            return jsonify({"error": "No tickers provided"}), 400

        cache_key = cache.make_cache_key(tickers, start, end, r_target, v_target, min_w, max_w)
        cached_bytes = cache.load(cache_key)

        if cached_bytes:
            return Response(cached_bytes, content_type='application/json')

        # --- Cache Miss: Compute Fresh Data ---
        returns_df, prices = data_service.get_returns(tickers, start, end)
        if returns_df.empty:
            return jsonify({"error": "No return data found for the given tickers and date range."}), 404

        mu = returns_df.loc[tickers, 'Annualized Return']
        bounds = [(min_w.get(t, 0.0), max_w.get(t, 0.2)) for t in mu.index]

        optimizer = PortfolioOptimizer()
        results = optimizer.run_optimizations(mu, prices, bounds, r_target, v_target)
        
        ef = results['Max Sharpe']
        vol = prices.pct_change().dropna().std() * (252 ** 0.5)
        sharpe = (mu - Config.RISK_FREE_RATE) / vol
        corr = PortfolioOptimizer.compute_corr_matrix(prices)
        perf, alloc = optimizer.compile_results(results, Config.RISK_FREE_RATE)
        chart = Chart(data_service)

        data_ef, layout_ef = chart.plot_efficient_frontier(ef, mu, vol)
        data_heatmap, layout_heatmap = chart.heatmap(mu.index.tolist())
        navs = data_service.compute_navs_for_strategies(alloc, start, end, 'monthly')
        data_port, layout_port = chart.plot_portfolios(navs, 'monthly')
        
        result_data = {
            'mu': mu.to_dict(),
            'vol': vol.to_dict(),
            'sharpe': sharpe.to_dict(),
            'perf_dict': perf,
            'alloc_dict': alloc,
            'corr_matrix': corr,
            'efficient_frontier_data': data_ef,
            'efficient_frontier_layout': layout_ef,
            'heatmap_data': data_heatmap,
            'heatmap_layout': layout_heatmap,
            'portfolio_data': {'monthly': data_port},
            'portfolio_layout': {'monthly': layout_port}
        }
        
        result_bytes = orjson.dumps(result_data, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        cache.save(cache_key, result_bytes)

        return Response(result_bytes, content_type='application/json')

    except Exception as e:
        app.logger.error(f"API Error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# --- Other API Routes ---
@app.route('/api/tickers')
def tickers_api():
    q = request.args.get('q', '').upper()
    matches = [t for t in data_service.get_tickers if t.startswith(q)]
    return Response(orjson.dumps(matches[:20]), content_type="application/json")

# @app.route('/export_weights', methods=['POST'])
# def export_weights():
#     # This route remains mostly the same, but it should rely on cached data
#     try:
#         data = request.form
#         selected = data.get("selected_tickers", "").split(",")
#         start, end = data.get('start_date'), data.get('end_date')
#         r_target = float(data.get('target_return', 0.05))
#         v_target = float(data.get('target_volatility', 0.10))
#         rebalance = data.get('rebalance')
        
#         # Reconstruct weights to find the correct cache key
#         # Note: This assumes weights are passed from the form, which might need adjustment
#         # in the JS to ensure they are sent with the export request.
#         min_w = {t: float(data.get(f"min_{i}", 0)) / 100 for i, t in enumerate(selected)}
#         max_w = {t: float(data.get(f"max_{i}", 15)) / 100 for i, t in enumerate(selected)}

#         cache_key = cache.make_cache_key(selected, start, end, r_target, v_target, min_w, max_w)
#         cached_bytes = cache.load(cache_key)
        
#         if not cached_bytes:
#             return "Error: No cached data found for these parameters. Please run the visualization first.", 404

#         cached = orjson.loads(cached_bytes)
#         alloc_dict = cached.get('alloc_dict')
#         weights = data_service.compute_weights_for_strategies(alloc_dict, start, end, rebalance=rebalance)

#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             for strat, df in weights.items():
#                 df.to_excel(writer, sheet_name=re.sub(r'[:\\/?*\[\]]', '', strat)[:31], index=False)

#         output.seek(0)
#         filename = f"weights_{rebalance}_{start}_to_{end}.xlsx"
#         return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#     except Exception as e:
#         app.logger.error(f"Export failed: {e}")
#         return f"Export failed: {e}", 500

@app.route('/health')
def health():
    return 'OK', 200

# This is for running locally, e.g., `python app.py`
if __name__ == '__main__':
    app.run(debug=True)