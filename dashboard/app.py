"""
app.py

Streamlit dashboard for real-time monitoring and control of pairs trading system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly. graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path. append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pairs_strategy import PairsTradingStrategy
from src. backtester import KalmanPairsBacktest
from src.data_manager import download_data
from src.pair_selection import PairSelector
from src. parameter_optimizer import ParameterOptimizer
from src.portfolio_optimizer import MultiPairPortfolio

# Page configuration
st.set_page_config(
    page_title="Kalman Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    . metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state."""
    
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'strategy' not in st.session_state:
            st.session_state. strategy = None
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'top_pairs' not in st.session_state:
            st.session_state.top_pairs = None


def main():
    """Main dashboard function."""
    
    # Initialize state
    DashboardState.initialize()
    
    # Header
    st.markdown('<p class="main-header">📈 Kalman Filter Pairs Trading Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Pairs+Trading", 
                use_column_width=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Backtest", "🔍 Pair Selection", "⚙️ Optimization", 
             "📈 Live Trading", "📖 Documentation"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        if st.session_state.backtest_results is not None:
            results = st.session_state.backtest_results
            total_return = (results['cumulative_return']. iloc[-1] - 1) * 100
            
            col1, col2 = st. columns(2)
            with col1:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col2:
                returns = results['strategy_return'].dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                st. metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Backtest": 
        show_backtest_page()
    elif page == "🔍 Pair Selection":
        show_pair_selection_page()
    elif page == "⚙️ Optimization":
        show_optimization_page()
    elif page == "📈 Live Trading":
        show_live_trading_page()
    elif page == "📖 Documentation": 
        show_documentation_page()


def show_home_page():
    """Display home page."""
    st.title("Welcome to Kalman Pairs Trading System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - Dynamic hedge ratio estimation
        - Real-time risk management
        - Multi-pair optimization
        - Automated pair selection
        - Machine learning integration
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Performance
        - Backtesting engine
        - Walk-forward analysis
        - Monte Carlo simulation
        - Parameter optimization
        - Statistical testing
        """)
    
    with col3:
        st. markdown("""
        ### 🚀 Trading
        - Live trading support
        - Broker integration
        - Position monitoring
        - Risk controls
        - Alert system
        """)
    
    st.markdown("---")
    
    # Quick start
    st.subheader("🚀 Quick Start")
    
    col1, col2 = st. columns(2)
    
    with col1:
        st. markdown("""
        **1. Select a Pair**
        - Use Pair Selection to find cointegrated pairs
        - Or manually input tickers
        
        **2. Download Data**
        - Specify date range
        - System downloads and cleans data
        
        **3. Run Backtest**
        - Configure strategy parameters
        - View comprehensive results
        """)
    
    with col2:
        st.markdown("""
        **4. Optimize Parameters**
        - Grid search or Bayesian optimization
        - Walk-forward analysis
        
        **5. Deploy Live**
        - Connect to broker
        - Monitor positions
        - Automated execution
        """)
    
    # Recent activity
    st.markdown("---")
    st.subheader("📈 Recent Activity")
    
    if st.session_state.backtest_results is not None:
        results = st.session_state. backtest_results
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (results['cumulative_return'].iloc[-1] - 1) * 100
            st.metric("Total Return", f"{total_return:.2f}%")
        
        with col2:
            returns = results['strategy_return'].dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            cumulative = results['cumulative_return']
            running_max = cumulative.cummax()
            drawdown = ((cumulative - running_max) / running_max * 100).min()
            st.metric("Max Drawdown", f"{drawdown:.2f}%")
        
        with col4:
            n_trades = len(results[results['action']. str.contains('EXIT', na=False)])
            st.metric("Number of Trades", n_trades)
        
        # Quick chart
        fig = go.Figure()
        fig.add_trace(go. Scatter(
            x=results. index,
            y=results['cumulative_return'],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Recent Backtest Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent backtests.  Run a backtest to see results here.")


def show_backtest_page():
    """Display backtest page."""
    st.title("📊 Strategy Backtesting")
    
    # Data input
    st.subheader("1️⃣ Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker_a = st.text_input("Asset A Ticker", value="GLD", key="ticker_a")
        start_date = st.date_input("Start Date", value=pd. to_datetime("2020-01-01"))
    
    with col2:
        ticker_b = st.text_input("Asset B Ticker", value="GDX", key="ticker_b")
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    
    if st.button("📥 Download Data", key="download_data"):
        with st.spinner("Downloading data... "):
            try:
                data = download_data(
                    ticker_a, ticker_b,
                    start_date. strftime('%Y-%m-%d'),
                    end_date. strftime('%Y-%m-%d')
                )
                st. session_state.data = data
                st.success(f"Downloaded {len(data)} observations")
                
                # Show data preview
                st.dataframe(data.tail(10))
                
            except Exception as e:
                st. error(f"Error downloading data:  {e}")
    
    # Strategy parameters
    st.markdown("---")
    st.subheader("2️⃣ Strategy Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry_zscore = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
        exit_zscore = st.slider("Exit Z-Score", 0.1, 1.0, 0.5, 0.1)
    
    with col2:
        stop_loss_zscore = st.slider("Stop Loss Z-Score", 3.0, 6.0, 4.0, 0.5)
        lookback_window = st.slider("Lookback Window", 10, 50, 20, 5)
    
    with col3:
        delta = st.select_slider("Delta (Process Noise)", 
                                 options=[1e-5, 1e-4, 1e-3, 1e-2],
                                 value=1e-4,
                                 format_func=lambda x: f"{x:.0e}")
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
    
    # Run backtest
    if st.button("🚀 Run Backtest", key="run_backtest"):
        if st.session_state.data is None:
            st.error("Please download data first!")
        else:
            with st.spinner("Running backtest..."):
                try:
                    # Create strategy
                    strategy = PairsTradingStrategy(
                        entry_zscore=entry_zscore,
                        exit_zscore=exit_zscore,
                        stop_loss_zscore=stop_loss_zscore,
                        lookback_window=lookback_window,
                        delta=delta,
                        transaction_cost=transaction_cost
                    )
                    
                    # Run backtest
                    backtester = KalmanPairsBacktest(strategy)
                    results = backtester.run(st.session_state.data)
                    
                    # Store results
                    st.session_state.backtest_results = results
                    st.session_state.strategy = strategy
                    
                    st.success("Backtest completed!")
                    
                except Exception as e:
                    st.error(f"Error running backtest: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.backtest_results is not None:
        st.markdown("---")
        st.subheader("3️⃣ Results")
        
        results = st.session_state.backtest_results
        
        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        returns = results['strategy_return'].dropna()
        total_return = (results['cumulative_return'].iloc[-1] - 1) * 100
        annual_return = ((1 + total_return/100) ** (252 / len(returns)) - 1) * 100
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = results['cumulative_return']
        running_max = cumulative. cummax()
        drawdown = ((cumulative - running_max) / running_max * 100).min()
        
        n_trades = len(results[results['action'].str.contains('EXIT', na=False)])
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Annual Return", f"{annual_return:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{drawdown:.2f}%")
        with col5:
            st.metric("Trades", n_trades)
        
        # Charts
        st.markdown("### 📈 Performance Charts")
        
        tabs = st.tabs(["Overview", "Spread Analysis", "Beta Evolution", "Trade Analysis"])
        
        with tabs[0]:
            # Cumulative returns and drawdown
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Cumulative Returns", "Drawdown"),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Cumulative returns
            fig. add_trace(
                go.Scatter(x=results.index, y=results['cumulative_return'],
                          mode='lines', name='Strategy',
                          line=dict(color='#1f77b4', width=2)),
                row=1, col=1
            )
            
            # Drawdown
            drawdown_series = (cumulative - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(x=results. index, y=drawdown_series,
                          mode='lines', name='Drawdown',
                          fill='tozeroy', line=dict(color='red', width=1)),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_layout(height=600, showlegend=False, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            # Spread and z-score
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Spread", "Z-Score with Signals"),
                vertical_spacing=0.15
            )
            
            # Spread
            fig.add_trace(
                go.Scatter(x=results.index, y=results['spread'],
                          mode='lines', name='Spread',
                          line=dict(color='purple', width=1)),
                row=1, col=1
            )
            
            # Z-score
            fig.add_trace(
                go.Scatter(x=results. index, y=results['zscore'],
                          mode='lines', name='Z-Score',
                          line=dict(color='blue', width=1)),
                row=2, col=1
            )
            
            # Entry/exit thresholds
            fig.add_hline(y=entry_zscore, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=1)
            fig.add_hline(y=-entry_zscore, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                         opacity=0.3, row=2, col=1)
            
            # Add trade signals
            long_entries = results[results['action'] == 'LONG_SPREAD']
            short_entries = results[results['action'] == 'SHORT_SPREAD']
            exits = results[results['action']. str.contains('EXIT', na=False)]
            
            if len(long_entries) > 0:
                fig.add_trace(
                    go. Scatter(x=long_entries. index, y=long_entries['zscore'],
                              mode='markers', name='Long Entry',
                              marker=dict(color='green', size=10, symbol='triangle-up')),
                    row=2, col=1
                )
            
            if len(short_entries) > 0:
                fig.add_trace(
                    go.Scatter(x=short_entries.index, y=short_entries['zscore'],
                              mode='markers', name='Short Entry',
                              marker=dict(color='red', size=10, symbol='triangle-down')),
                    row=2, col=1
                )
            
            if len(exits) > 0:
                fig.add_trace(
                    go. Scatter(x=exits.index, y=exits['zscore'],
                              mode='markers', name='Exit',
                              marker=dict(color='blue', size=8, symbol='x')),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Spread", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=2, col=1)
            fig.update_layout(height=600, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            # Beta evolution
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results.index, y=results['beta'],
                mode='lines', name='Dynamic Beta',
                line=dict(color='green', width=2)
            ))
            
            # Add rolling mean
            beta_ma = results['beta'].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=results.index, y=beta_ma,
                mode='lines', name='20-Day MA',
                line=dict(color='orange', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title="Hedge Ratio Evolution",
                xaxis_title="Date",
                yaxis_title="Beta",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Beta statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Beta", f"{results['beta']. mean():.4f}")
            with col2:
                st.metric("Std Beta", f"{results['beta']. std():.4f}")
            with col3:
                st. metric("Beta Range", 
                         f"{results['beta'].min():.4f} - {results['beta'].max():.4f}")
        
        with tabs[3]:
            # Trade analysis
            trades_df = results[results['action']. str.contains('LONG_SPREAD|SHORT_SPREAD|EXIT', na=False)].copy()
            
            if len(trades_df) > 0:
                st.markdown("### Recent Trades")
                st.dataframe(
                    trades_df[['action', 'zscore', 'spread', 'beta', 'strategy_return']].tail(20),
                    use_container_width=True
                )
                
                # Trade distribution
                col1, col2 = st. columns(2)
                
                with col1:
                    # Returns distribution
                    trade_returns = results[results['action'].str.contains('EXIT', na=False)]['strategy_return']
                    
                    fig = go.Figure(data=[go.Histogram(x=trade_returns * 100, nbinsx=30)])
                    fig.update_layout(
                        title="Trade Returns Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Win/loss analysis
                    wins = len(trade_returns[trade_returns > 0])
                    losses = len(trade_returns[trade_returns <= 0])
                    
                    fig = go.Figure(data=[go. Pie(
                        labels=['Wins', 'Losses'],
                        values=[wins, losses],
                        hole=. 3,
                        marker_colors=['#2ecc71', '#e74c3c']
                    )])
                    fig.update_layout(title="Win/Loss Ratio", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades executed yet.")


def show_pair_selection_page():
    """Display pair selection page."""
    st.title("🔍 Automated Pair Selection")
    
    st.markdown("""
    Find the best pairs for trading using statistical cointegration tests.
    """)
    
    # Universe input
    st.subheader("Define Universe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        universe_input = st.text_area(
            "Enter tickers (one per line)",
            value="SPY\nQQQ\nIWM\nDIA\nGLD\nSLV\nTLT\nXLE\nXLF\nXLK",
            height=200
        )
        
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"), key="pair_start")
    
    with col2:
        min_correlation = st.slider("Min Correlation", 0.0, 1.0, 0.5, 0.05)
        max_cointegration_pvalue = st.slider("Max Cointegration P-Value", 0.01, 0.10, 0.05, 0.01)
        max_half_life = st.slider("Max Half-Life (days)", 10, 120, 60, 10)
        
        end_date = st.date_input("End Date", value=pd. to_datetime("2024-01-01"), key="pair_end")
    
    top_n = st.slider("Number of top pairs to return", 5, 50, 10, 5)
    
    # Scan universe
    if st.button("🔎 Scan Universe", key="scan_universe"):
        universe = [ticker.strip() for ticker in universe_input.split('\n') if ticker.strip()]
        
        if len(universe) < 2:
            st.error("Please enter at least 2 tickers")
        else:
            with st.spinner(f"Scanning {len(universe)} assets for pairs..."):
                try:
                    selector = PairSelector(
                        min_correlation=min_correlation,
                        max_cointegration_pvalue=max_cointegration_pvalue,
                        max_half_life=max_half_life
                    )
                    
                    top_pairs = selector.scan_universe(
                        universe,
                        start_date=start_date. strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        top_n=top_n
                    )
                    
                    st.session_state.top_pairs = top_pairs
                    st.success(f"Found {len(top_pairs)} suitable pairs!")
                    
                except Exception as e:
                    st.error(f"Error scanning universe: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.top_pairs is not None:
        st.markdown("---")
        st.subheader("Top Pairs")
        
        pairs = st.session_state.top_pairs
        
        # Create DataFrame
        pairs_df = pd.DataFrame([{
            'Pair': f"{p. asset_a}/{p.asset_b}",
            'Score': f"{p.total_score:.4f}",
            'Cointegration P-Value': f"{p.cointegration_pvalue:.4f}",
            'Correlation': f"{p.correlation:. 4f}",
            'Half-Life': f"{p.half_life:. 2f}",
            'Hurst':  f"{p.hurst_exponent:.4f}"
        } for p in pairs])
        
        st.dataframe(pairs_df, use_container_width=True)
        
        # Visualization
        col1, col2 = st. columns(2)
        
        with col1:
            # Scores bar chart
            fig = go. Figure(data=[
                go.Bar(
                    x=[f"{p.asset_a}/{p.asset_b}" for p in pairs],
                    y=[p.total_score for p in pairs],
                    marker_color='steelblue'
                )
            ])
            fig.update_layout(
                title="Pair Scores",
                xaxis_title="Pair",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter:  Half-life vs P-value
            fig = go. Figure(data=[
                go. Scatter(
                    x=[p.half_life for p in pairs],
                    y=[p.cointegration_pvalue for p in pairs],
                    mode='markers+text',
                    text=[f"{p.asset_a}/{p.asset_b}" for p in pairs],
                    textposition="top center",
                    marker=dict(
                        size=[p.total_score * 100 for p in pairs],
                        color=[p.total_score for p in pairs],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Score")
                    )
                )
            ])
            fig.add_hline(y=max_cointegration_pvalue, line_dash="dash", 
                         line_color="red", opacity=0.5)
            fig.update_layout(
                title="Pair Quality Landscape",
                xaxis_title="Half-Life (days)",
                yaxis_title="Cointegration P-Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Select pair for backtesting
        st.markdown("---")
        selected_pair = st.selectbox(
            "Select pair for backtesting",
            options=[f"{p.asset_a}/{p.asset_b}" for p in pairs]
        )
        
        if st.button("Load Pair for Backtesting"):
            asset_a, asset_b = selected_pair.split('/')
            st.session_state.ticker_a = asset_a
            st.session_state.ticker_b = asset_b
            st.success(f"Loaded {selected_pair}.  Go to Backtest page to continue.")


def show_optimization_page():
    """Display optimization page."""
    st.title("⚙️ Parameter Optimization")
    
    st.markdown("""
    Optimize strategy parameters using various methods.
    """)
    
    if st.session_state.data is None:
        st.warning("Please download data in the Backtest page first!")
        return
    
    # Optimization method
    method = st.radio(
        "Optimization Method",
        ["Grid Search", "Random Search", "Bayesian Optimization", "Walk-Forward Analysis"]
    )
    
    st.markdown("---")
    
    if method == "Grid Search": 
        st.subheader("Grid Search Configuration")
        
        col1, col2 = st. columns(2)
        
        with col1:
            entry_values = st.multiselect(
                "Entry Z-Score Values",
                options=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                default=[1.5, 2.0, 2.5]
            )
            
            exit_values = st.multiselect(
                "Exit Z-Score Values",
                options=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                default=[0.3, 0.5, 0.7]
            )
        
        with col2:
            delta_values = st.multiselect(
                "Delta Values",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                default=[1e-5, 1e-4, 1e-3],
                format_func=lambda x: f"{x:.0e}"
            )
            
            lookback_values = st.multiselect(
                "Lookback Window Values",
                options=[10, 15, 20, 25, 30],
                default=[15, 20, 30]
            )
        
        objective = st.selectbox(
            "Objective Function",
            ["sharpe_ratio", "total_return", "calmar_ratio", "sortino_ratio"]
        )
        
        if st.button("Run Grid Search"):
            param_grid = {
                'entry_zscore':  entry_values,
                'exit_zscore': exit_values,
                'delta': delta_values,
                'lookback_window': lookback_values
            }
            
            n_combinations = np.prod([len(v) for v in param_grid. values()])
            
            with st.spinner(f"Testing {n_combinations} parameter combinations..."):
                try:
                    optimizer = ParameterOptimizer(
                        PairsTradingStrategy,
                        st.session_state.data,
                        objective_function=objective
                    )
                    
                    best_result = optimizer.grid_search(param_grid, verbose=False)
                    
                    st.success("Optimization complete!")
                    
                    # Display results
                    st.subheader("Best Parameters")
                    
                    col1, col2 = st. columns(2)
                    
                    with col1:
                        for param, value in best_result['params'].items():
                            if isinstance(value, float) and value < 0.01:
                                st.metric(param, f"{value:.2e}")
                            else:
                                st.metric(param, f"{value}")
                    
                    with col2:
                        st.metric(f"Best {objective}", f"{best_result['score']:.4f}")
                    
                    # Heatmap
                    if len(entry_values) > 1 and len(exit_values) > 1:
                        st.subheader("Parameter Heatmap")
                        
                        # Create pivot table
                        results_list = []
                        for result in optimizer.results:
                            results_list. append({
                                'entry_zscore': result['params']['entry_zscore'],
                                'exit_zscore': result['params']['exit_zscore'],
                                'score': result['score']
                            })
                        
                        results_df = pd.DataFrame(results_list)
                        pivot = results_df.pivot_table(
                            values='score',
                            index='exit_zscore',
                            columns='entry_zscore'
                        )
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=pivot. values,
                            x=pivot.columns,
                            y=pivot.index,
                            colorscale='RdYlGn',
                            text=pivot.values,
                            texttemplate='%{text:.3f}',
                            textfont={"size": 10}
                        ))
                        
                        fig.update_layout(
                            title="Entry vs Exit Z-Score Performance",
                            xaxis_title="Entry Z-Score",
                            yaxis_title="Exit Z-Score",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during optimization: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def show_live_trading_page():
    """Display live trading page."""
    st.title("📈 Live Trading")
    
    st.warning("⚠️ Live trading involves real financial risk. Use paper trading for testing.")
    
    # Broker selection
    st.subheader("Broker Configuration")
    
    broker = st.selectbox("Select Broker", ["Alpaca (Paper)", "Alpaca (Live)", "Interactive Brokers"])
    
    if broker. startswith("Alpaca"):
        col1, col2 = st. columns(2)
        
        with col1:
            api_key = st.text_input("API Key", type="password")
        
        with col2:
            api_secret = st.text_input("API Secret", type="password")
        
        if st. button("Connect to Broker"):
            st.info("Live trading functionality requires broker credentials and is disabled in this demo.")
            st.markdown("""
            **To enable live trading:**
            1. Set up a broker account (Alpaca, Interactive Brokers, etc.)
            2. Obtain API credentials
            3. Configure credentials in the application
            4. Start the trading system
            
            **Features available in live mode:**
            - Real-time price streaming
            - Automated order execution
            - Position monitoring
            - Risk management
            - Alert notifications
            """)
    
    # Trading status (mock)
    st.markdown("---")
    st.subheader("Trading Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Account Value", "$100,000")
    with col2:
        st.metric("Open Positions", "0")
    with col3:
        st.metric("Today's P&L", "$0.00")
    with col4:
        st.metric("Total P&L", "$0.00")


def show_documentation_page():
    """Display documentation page."""
    st.title("📖 Documentation")
    
    tabs = st.tabs(["Overview", "Strategy", "Risk Management", "API Reference"])
    
    with tabs[0]:
        st.markdown("""
        ## Kalman Filter Pairs Trading
        
        ### What is Pairs Trading?
        
        Pairs trading is a market-neutral strategy that involves: 
        1. Identifying two cointegrated assets
        2. Trading the spread between them
        3. Profiting from mean reversion
        
        ### Why Kalman Filter?
        
        Traditional pairs trading assumes a static hedge ratio between assets. 
        The Kalman Filter provides:
        - **Dynamic hedge ratio** that adapts to market conditions
        - **Online learning** without retraining
        - **Noise filtering** for robust estimates
        - **Probabilistic framework** with confidence intervals
        
        ### System Components
        
        1. **Data Manager**: Download and clean price data
        2. **Pair Selector**: Find cointegrated pairs automatically
        3. **Kalman Filter**: Estimate time-varying hedge ratios
        4. **Strategy Engine**: Generate trading signals
        5. **Risk Manager**: Control position sizes and losses
        6. **Backtester**:  Evaluate historical performance
        7. **Optimizer**: Find best parameters
        8. **Live Trader**: Execute trades in real-time
        """)
    
    with tabs[1]:
        st.markdown("""
        ## Trading Strategy Logic
        
        ### Signal Generation
        
        ```python
        # 1. Update Kalman Filter
        beta, alpha, spread = kalman_filter. update(price_a, price_b)
        
        # 2. Calculate Z-Score
        zscore = (spread - spread_mean) / spread_std
        
        # 3. Generate Signal
        if zscore > entry_threshold:
            signal = SHORT_SPREAD  # Sell A, Buy B
        elif zscore < -entry_threshold:
            signal = LONG_SPREAD   # Buy A, Sell B
        elif abs(zscore) < exit_threshold:
            signal = EXIT
        ```
        
        ### Parameters
        
        - **entry_zscore**:  Threshold for opening positions (default: 2.0)
        - **exit_zscore**: Threshold for closing positions (default: 0.5)
        - **stop_loss_zscore**: Maximum loss threshold (default: 4.0)
        - **lookback_window**: Window for spread statistics (default: 20)
        - **delta**:  Kalman filter process noise (default: 1e-4)
        
        ### Position Sizing
        
        Position size is determined by:
        - Spread volatility
        - Account risk limits
        - Kelly criterion
        - Maximum position constraints
        """)
    
    with tabs[2]:
        st.markdown("""
        ## Risk Management
        
        ### Pre-Trade Checks
        
        1. **Correlation Check**: Assets must maintain minimum correlation
        2. **Capital Check**: Sufficient capital available
        3. **Position Limit**: Maximum number of concurrent positions
        4. **Volatility Check**: Spread volatility within acceptable range
        
        ### During Trade
        
        1. **Stop Loss**: Exit if spread diverges beyond threshold
        2. **Drawdown Limit**: Stop trading if losses exceed limit
        3. **Time Stop**: Close positions after maximum holding period
        
        ### Position Sizing Methods
        
        - **Fixed Fraction**: Fixed % of capital per trade
        - **Volatility-Based**: Size inversely proportional to volatility
        - **Kelly Criterion**: Optimize position size based on edge
        - **Risk Parity**: Equal risk contribution per position
        """)
    
    with tabs[3]: 
        st.markdown("""
        ## API Reference
        
        ### Core Classes
        
        #### KalmanFilterRegression
        ```python
        from src.kalman_filter import KalmanFilterRegression
        
        kf = KalmanFilterRegression(delta=1e-4, Ve=1e-3)
        beta, alpha, spread = kf. update(price_a, price_b)
        ```
        
        #### PairsTradingStrategy
        ```python
        from src.pairs_strategy import PairsTradingStrategy
        
        strategy = PairsTradingStrategy(
            entry_zscore=2.0,
            exit_zscore=0.5,
            delta=1e-4
        )
        
        signal, info = strategy.generate_signal(price_a, price_b, spread_series)
        ```
        
        #### KalmanPairsBacktest
        ```python
        from src.backtester import KalmanPairsBacktest
        
        backtester = KalmanPairsBacktest(strategy)
        results = backtester.run(data)
        metrics = backtester.get_performance_metrics(results)
        ```
        
        ### Utility Functions
        
        ```python
        # Download data
        from src.data_manager import download_data
        df = download_data('GLD', 'GDX', '2020-01-01', '2024-01-01')
        
        # Find pairs
        from src.pair_selection import PairSelector
        selector = PairSelector()
        top_pairs = selector.scan_universe(universe)
        
        # Optimize parameters
        from src.parameter_optimizer import ParameterOptimizer
        optimizer = ParameterOptimizer(strategy, data)
        best_params = optimizer.grid_search(param_grid)
        ```
        """)


if __name__ == "__main__": 
    main()