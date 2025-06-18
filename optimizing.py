from typing import Dict, List, Tuple, Union
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import cla
import pandas as pd
from pypfopt import plotting
import numpy as np

class PortfolioOptimizer:
    @staticmethod
    def compute_corr_matrix(prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        return returns.corr()
    
    @staticmethod
    def run_optimizations(
        mu: pd.Series, 
        prices: pd.DataFrame, 
        bounds: List[Tuple[float, float]],
        target_return: float,
        target_volatility: float,
    ) -> Tuple[Dict[str, Union[EfficientFrontier, Dict]], Union[str, None]]:
        """Run portfolio optimization strategies and return ef objects and HTML plot."""

        cov_matrix = CovarianceShrinkage(prices).ledoit_wolf()
        results = {}
        fig_html = None  # default value
        try:
            ef_max_sharpe = cla.CLA(mu, cov_matrix, weight_bounds=bounds)
            ef_max_sharpe.max_sharpe()
            fig = plotting.plot_efficient_frontier(ef_max_sharpe, interactive=True, show_assets=False)

            rets = mu.values
            risks = np.sqrt(np.diag(cov_matrix))
            tickers = mu.index.tolist()
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta'] * (len(tickers) // 7 + 1)

            # Add labeled scatter points
            fig.add_scatter(
                x=risks,
                y=rets,
                mode='markers+text',
                text=tickers,
                textposition='top center',
                marker=dict(color=colors, size=10, line=dict(width=1, color='black')),
                name='Assets',
                hovertemplate=(
                    "Ticker: %{text}<br>" +
                    "Return: %{y:.2%}<br>" +
                    "Risk: %{x:.2%}<extra></extra>"
                )
            )

            # Customize layout
            fig.update_layout(
                title="Efficient Frontier with Tickers",
                xaxis_title="Volatility",
                yaxis_title="Return",
                hovermode="closest"
            )

            # Save or return as HTML
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            results["Max Sharpe"] = ef_max_sharpe
        except Exception as e:
            print("Plotting error:", e)
            results["Max Sharpe"] = {"Error": str(e)}

        strategies = {
            "Min Volatility": lambda ef: ef.min_volatility(),
            f"Efficient Return (target: {target_return:.3f})": lambda ef: ef.efficient_return(target_return),
            f"Efficient Risk (target: {target_volatility:.3f})": lambda ef: ef.efficient_risk(target_volatility),
        }

        for name, strategy in strategies.items():
            try:
                ef = EfficientFrontier(mu.astype(float), cov_matrix, weight_bounds=bounds)
                strategy(ef)
                results[name] = ef
            except Exception as e:
                results[name] = {"Error": str(e)}
        
        return results, fig_html

    @staticmethod
    def extract_performance(ef: EfficientFrontier, risk_free_rate: float):
        """Extract portfolio performance metrics from ef."""
        expected_return, volatility, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        return {
            "Expected Return": expected_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Weights": ef.clean_weights()
        }

    @staticmethod
    def compile_results(efs: Dict[str, Union[EfficientFrontier, Dict]], risk_free_rate: float):
        """Compile performance and allocation tables from ef objects."""
        perf_rows = []
        alloc_rows = []

        for strategy, ef_or_error in efs.items():
            if isinstance(ef_or_error, dict) and "Error" in ef_or_error:
                continue

            perf = PortfolioOptimizer.extract_performance(ef_or_error, risk_free_rate)
            perf_rows.append({
                "Strategy": strategy,
                "Expected Return": perf["Expected Return"],
                "Volatility": perf["Volatility"],
                "Sharpe Ratio": perf["Sharpe Ratio"]
            })

            alloc_row = {"Strategy": strategy}
            alloc_row.update(perf["Weights"])
            alloc_rows.append(alloc_row)

        return pd.DataFrame(perf_rows), pd.DataFrame(alloc_rows)
