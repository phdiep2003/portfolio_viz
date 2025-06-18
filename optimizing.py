from typing import Dict, List, Tuple, Union
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
import pandas as pd

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
        risk_free_rate: float,
        target_return: float,
        target_volatility: float,
    ) -> Dict[str, Union[EfficientFrontier, Dict]]:
        """Run portfolio optimization strategies, return ef objects or error info."""
        cov_matrix = CovarianceShrinkage(prices).ledoit_wolf()
        results = {}

        strategies = {
            "Min Volatility": lambda ef: ef.min_volatility(),
            "Max Sharpe": lambda ef: ef.max_sharpe(risk_free_rate=risk_free_rate),
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

        return results

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
