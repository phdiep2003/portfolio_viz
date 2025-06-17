import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from data_service import ParquetDataService

class Chart:
    @staticmethod
    def create_efficient_frontier_plot(mu, vol, sharpe, ef_returns, ef_volatility):
        """
        Create interactive efficient frontier plot with Plotly Express
        
        Parameters:
        - mu: Expected returns of individual assets
        - vol: Volatility of individual assets
        - sharpe: Sharpe ratios of individual assets
        - ef_returns: Efficient frontier portfolio returns
        - ef_volatility: Efficient frontier portfolio volatilities
        """
        # Create DataFrame for individual assets
        assets_df = pd.DataFrame({
            'Ticker': mu.index,
            'Expected Return': mu * 100,
            'Volatility': vol * 100,
            'Sharpe Ratio': sharpe
        })
        
        # Sort frontier points by volatility to ensure proper line drawing
        frontier_points = sorted(zip(ef_volatility * 100, ef_returns * 100), key=lambda x: x[0])
        frontier_vol, frontier_ret = zip(*frontier_points)
        
        # Create base figure with individual assets
        fig = go.Figure()
        
        # Add individual assets with color based on Sharpe ratio (but without colorbar)
        for _, row in assets_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row['Volatility']],
                    y=[row['Expected Return']],
                    mode='markers+text',
                    name=row['Ticker'],
                    marker=dict(
                        size=12,
                        color=row['Sharpe Ratio'],
                        colorscale='Viridis',
                        showscale=False,  # This removes the colorbar
                        line=dict(width=1, color='black')
                    ),
                    text=[row['Ticker']],
                    textposition='top center',
                    hovertemplate=
                        '<b>'+row['Ticker']+'</b><br>'+
                        'Volatility: %{x:.2f}%<br>' +
                        'Return: %{y:.2f}%<br>' +
                        'Sharpe: %.2f<extra></extra>' % row['Sharpe Ratio'],
                    showlegend=False
                )
            )
        
        # Add efficient frontier line (now properly sorted)
        fig.add_trace(
            go.Scatter(
                x=frontier_vol,
                y=frontier_ret,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='red', width=3),
                hovertemplate='<b>Volatility</b>: %{x:.2f}%<br><b>Return</b>: %{y:.2f}%'
            )
        )
        
        # Highlight max Sharpe ratio portfolio
        max_sharpe_idx = np.argmax((ef_returns - 0.02) / ef_volatility)  # Assuming 2% risk-free rate
        fig.add_trace(
            go.Scatter(
                x=[ef_volatility[max_sharpe_idx] * 100],
                y=[ef_returns[max_sharpe_idx] * 100],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='green',
                    line=dict(width=1, color='black')
                ),
                hovertemplate=
                    '<b>Max Sharpe Portfolio</b><br>' +
                    'Volatility: %{x:.2f}%<br>' +
                    'Return: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Add minimum volatility portfolio (leftmost point)
        min_vol_idx = np.argmin(ef_volatility)
        fig.add_trace(
            go.Scatter(
                x=[ef_volatility[min_vol_idx] * 100],
                y=[ef_returns[min_vol_idx] * 100],
                mode='markers',
                name='Min Volatility',
                marker=dict(
                    symbol='diamond',
                    size=20,
                    color='orange',
                    line=dict(width=1, color='black')
                ),
                hovertemplate=
                    '<b>Min Volatility Portfolio</b><br>' +
                    'Volatility: %{x:.2f}%<br>' +
                    'Return: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Annualized Return (%)',
            hovermode='closest',
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    @staticmethod
    def heatmap(selected_tickers):
        data_service = ParquetDataService()
        ticker_sector_df = data_service.tickers_with_sectors
        filtered_df = ticker_sector_df[ticker_sector_df['Ticker'].isin(selected_tickers)].copy()
        
        # Create equal weights for uniform box sizes
        filtered_df['Weight'] = 1
        
        # Create color mapping for sectors only
        unique_sectors = filtered_df['Sector'].unique()
        sector_colors = px.colors.qualitative.G10[:len(unique_sectors)]
        color_map = dict(zip(unique_sectors, sector_colors))
        
        # Create the treemap with all boxes initially white
        fig = px.treemap(
            filtered_df,
            path=['Sector', 'Ticker'],
            values='Weight',
            color_discrete_sequence=['white']  # Start with all white
        )
        
        # Custom coloring - only color the sector (parent) boxes
        colors = []
        text_colors = []
        for label, parent in zip(fig.data[0].labels, fig.data[0].parents):
            if parent == '':  # This is a sector box
                colors.append(color_map[label])
                text_colors.append('white')  # White text for colored sectors
            else:  # This is a stock box
                colors.append('white')
                text_colors.append('black')  # Black text for white stocks
        
        # Apply custom styling
        fig.update_traces(
            marker_colors=colors,
            textfont_color=text_colors,
            textinfo="label",
            textposition="middle center",
            textfont=dict(size=16),
            marker=dict(
                line=dict(width=1, color="darkgray")
            ),
            hovertemplate='<b>%{label}</b><br>Sector: %{parent}<extra></extra>',
            branchvalues='total',
            tiling=dict(
                packing='squarify',
                squarifyratio=1  # More uniform box sizes
            )
        )
        
        fig.update_layout(
            margin=dict(t=30, l=0, r=0, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            uniformtext=dict(
                minsize=12,
                mode='hide'
            ),
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn') 