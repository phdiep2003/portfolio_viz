import plotly.express as px
import plotly.graph_objects as go
from data_service import ParquetDataService
from pypfopt import plotting
import pandas as pd
import numpy as np

class Chart:
    def __init__(self, data_service=None):
        self.data_service = data_service or ParquetDataService()

    def convert_ndarrays(self, obj):
        """Recursively convert all numpy.ndarray in obj to lists."""
        if isinstance(obj, dict):
            return {k: self.convert_ndarrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarrays(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_ndarrays(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def plot_efficient_frontier(self, ef_max_sharpe, mu: pd.Series, vol: pd.Series):
        fig = plotting.plot_efficient_frontier(ef_max_sharpe, interactive=True, show_assets=False)

        tickers = mu.index.tolist()
        rets = mu.values
        risks = vol.loc[tickers].values

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta'] * (len(tickers) // 7 + 1)

        fig.add_scatter(
            x=risks,
            y=rets,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            marker=dict(color=colors[:len(tickers)], size=10, line=dict(width=1, color='black')),
            name='Assets',
            hovertemplate=(
                "Ticker: %{text}<br>" +
                "Return: %{y:.2%}<br>" +
                "Risk: %{x:.2%}<extra></extra>"
            )
        )

        fig.update_layout(
            title="Efficient Frontier with Asset Tickers",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            hovermode="closest",
            template='plotly_white',
            height=500,
            autosize=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,           # push legend slightly above the plot area
                xanchor="right",
                x=1
            ),
            margin=dict(t=100)   # increase top margin to avoid overlap
        )

        data = [self.convert_ndarrays(trace.to_plotly_json()) for trace in fig.data]
        layout = self.convert_ndarrays(fig.layout.to_plotly_json())
        return data, layout

    def heatmap(self, selected_tickers):
        ticker_sector_df = self.data_service.tickers_with_sectors
        filtered_df = ticker_sector_df[ticker_sector_df['Ticker'].isin(selected_tickers)].copy()
        filtered_df['Weight'] = 1

        unique_sectors = filtered_df['Sector'].unique()
        sector_colors = px.colors.qualitative.G10[:len(unique_sectors)]
        color_map = dict(zip(unique_sectors, sector_colors))

        fig = px.treemap(filtered_df, path=['Sector', 'Ticker'], values='Weight', color_discrete_sequence=['white'])

        colors = []
        text_colors = []
        for label, parent in zip(fig.data[0].labels, fig.data[0].parents):
            if parent == '':
                colors.append(color_map[label])
                text_colors.append('white')
            else:
                colors.append('white')
                text_colors.append('black')

        fig.update_traces(
            marker_colors=colors,
            textfont_color=text_colors,
            textinfo="label",
            textposition="middle center",
            textfont=dict(size=16),
            marker=dict(line=dict(width=1, color="darkgray")),
            hovertemplate='<b>%{label}</b><br>Sector: %{parent}<extra></extra>',
            branchvalues='total',
            tiling=dict(packing='squarify', squarifyratio=1)
        )

        fig.update_layout(
            margin=dict(t=30, l=0, r=0, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            uniformtext=dict(minsize=12, mode='hide'),
            showlegend=False
        )

        data = [self.convert_ndarrays(trace.to_plotly_json()) for trace in fig.data]
        layout = self.convert_ndarrays(fig.layout.to_plotly_json())
        return data, layout
    
    def plot_portfolios(self, navs: dict[str, pd.Series], rebalance: str = 'monthly'):
        fig = go.Figure()

        for name, nav in navs.items():
            # Drop NaNs to avoid length mismatch or invisible lines
            nav_clean = nav.dropna()
            if nav_clean.empty:
                continue

            x_vals = nav_clean.index.strftime('%Y-%m-%d').tolist()
            y_vals = nav_clean.values.tolist()

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=name,
                hoverinfo='skip',
                hovertemplate=None
            ))

        fig.update_layout(
            title=f'Portfolio Performance ({rebalance.capitalize()} Rebalancing)',
            template='plotly_white',
            height=500,
            autosize=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.96,
                xanchor="right",
                x=1
            ),
            xaxis=dict(title="Time"),
            yaxis=dict(title="Return (%)", ticksuffix="%", showgrid=True)
        )

        # Serialize into Plotly-react compatible format
        data_json = [self.convert_ndarrays(trace.to_plotly_json()) for trace in fig.data]
        layout_json = self.convert_ndarrays(fig.layout.to_plotly_json())

        return data_json, layout_json
