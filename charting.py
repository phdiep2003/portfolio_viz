import plotly.express as px
import plotly.graph_objects as go
from data_service import ParquetDataService
from pypfopt import plotting
import pandas as pd

class Chart:
    def __init__(self, data_service=None):
        self.data_service = data_service or ParquetDataService()

    # @staticmethod
    def plot_efficient_frontier(self, ef_max_sharpe, mu: pd.Series, vol: pd.Series) -> str:
        # Base efficient frontier figure without assets
        fig = plotting.plot_efficient_frontier(ef_max_sharpe, interactive=True, show_assets=False)

        # Prepare tickers, returns, risks
        tickers = mu.index.tolist()
        rets = mu.values
        risks = vol.loc[tickers].values

        # Color palette
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta'] * (len(tickers) // 7 + 1)

        # Add asset points with labels
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
            height=500
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    
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

        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def plot_portfolios(self, navs, rebalance='monthly'):
        fig = go.Figure()

        for name, nav in navs.items():
            fig.add_trace(go.Scatter(
                x=nav.index,
                y=nav,
                mode='lines',
                name=name,
                hoverinfo='skip',        # Disable hover info
                hovertemplate=None
            ))

        fig.update_layout(
            title=f'Portfolio Performance ({rebalance.capitalize()} Rebalancing)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value (Normalized to 100%)',
            template='plotly_white',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)