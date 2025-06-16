# Portfolio Visualization and Optimization

This project is a Flask-based web app for portfolio analysis, including:

- Loading historical price data from Parquet files.
- Portfolio optimization with per-asset weight bounds.
- Visualization of expected returns, performance metrics, asset allocation, and correlation matrices.
- Dynamic ticker autocomplete via API.


## Setup and Usage

1. Create and activate a Python virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask app:

    ```bash
    flask run
    ```

4. Access the web app at `http://localhost:5000`

## Notes

- Cached data files in `cache/` are ignored from Git and help speed up repeated calculations.
- The `/api/tickers` endpoint provides dynamic autocomplete for ticker symbols.
- Customize the asset count and weights in the form on the main page.

## License

[MIT License](LICENSE)


