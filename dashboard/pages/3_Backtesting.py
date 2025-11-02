import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
sys.path.append('../..')

from src.loaders.database import DatabaseManager
from src.config.settings import settings

st.set_page_config(page_title="Backtesting", page_icon="🧪", layout="wide")

# Initialize
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()

st.title("🧪 Strategy Backtesting")
st.markdown("Test and evaluate trading strategies using historical data")

# Sidebar - Strategy Configuration
st.sidebar.header("⚙️ Strategy Configuration")

strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    ["RSI Strategy", "Moving Average Crossover", "MACD Strategy", "Bollinger Bands"]
)

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    settings.NIFTY_50_STOCKS
)

initial_capital = st.sidebar.number_input(
    "Initial Capital (₹)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000
)

# Strategy Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Strategy Parameters")

if strategy_type == "RSI Strategy":
    rsi_buy = st.sidebar.slider("RSI Buy Threshold (Oversold)", 20, 40, 30)
    rsi_sell = st.sidebar.slider("RSI Sell Threshold (Overbought)", 60, 80, 70)
    rsi_period = st.sidebar.slider("RSI Period", 10, 20, 14)
    
elif strategy_type == "Moving Average Crossover":
    ma_short = st.sidebar.slider("Short MA Period", 5, 50, 20)
    ma_long = st.sidebar.slider("Long MA Period", 50, 200, 50)
    
elif strategy_type == "MACD Strategy":
    st.sidebar.info("Using standard MACD parameters (12, 26, 9)")
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
elif strategy_type == "Bollinger Bands":
    bb_period = st.sidebar.slider("BB Period", 10, 30, 20)
    bb_std = st.sidebar.slider("BB Std Deviation", 1.5, 3.0, 2.0, 0.1)

# Date Range
st.sidebar.markdown("---")
date_range = st.sidebar.selectbox(
    "Backtest Period",
    ["3 Months", "6 Months", "1 Year", "2 Years"],
    index=2
)

range_mapping = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
days = range_mapping[date_range]

# Load Data
@st.cache_data(ttl=300)
def load_data(symbol, days):
    price_df = db.get_latest_stock_data(symbol, days)
    indicators_df = db.get_latest_indicators(symbol, days)
    
    if price_df.empty or indicators_df.empty:
        return pd.DataFrame()
    
    merged_df = price_df.merge(indicators_df, on=['symbol', 'date'], how='inner')
    return merged_df.sort_values('date').reset_index(drop=True)

df = load_data(selected_stock, days)

if df.empty:
    st.error(f"No data available for {selected_stock}. Please run the data pipeline first.")
    st.stop()

# Backtesting Functions
def backtest_rsi_strategy(df, buy_threshold, sell_threshold, initial_capital):
    """RSI-based trading strategy"""
    df = df.copy().reset_index(drop=True)
    df['position'] = 0
    df['cash'] = float(initial_capital)
    df['holdings'] = 0.0
    df['total'] = float(initial_capital)
    
    position = 0
    cash = float(initial_capital)
    shares = 0
    trades = []
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'rsi_14']):
            df.loc[i, 'position'] = position
            df.loc[i, 'cash'] = cash
            df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
            df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
            continue
        
        # Buy signal
        if df.loc[i, 'rsi_14'] < buy_threshold and position == 0 and cash > 0:
            shares = int(cash / df.loc[i, 'close'])
            if shares > 0:
                cost = shares * df.loc[i, 'close']
                cash -= cost
                position = 1
                trades.append({
                    'date': df.loc[i, 'date'],
                    'type': 'BUY',
                    'price': df.loc[i, 'close'],
                    'shares': shares,
                    'value': cost
                })
        
        # Sell signal
        elif df.loc[i, 'rsi_14'] > sell_threshold and position == 1 and shares > 0:
            revenue = shares * df.loc[i, 'close']
            cash += revenue
            trades.append({
                'date': df.loc[i, 'date'],
                'type': 'SELL',
                'price': df.loc[i, 'close'],
                'shares': shares,
                'value': revenue
            })
            shares = 0
            position = 0
        
        df.loc[i, 'position'] = position
        df.loc[i, 'cash'] = cash
        df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
        df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
    
    return df, trades

def backtest_ma_crossover(df, short_period, long_period, initial_capital):
    """Moving Average Crossover strategy"""
    df = df.copy().reset_index(drop=True)
    df['position'] = 0
    df['cash'] = float(initial_capital)
    df['holdings'] = 0.0
    df['total'] = float(initial_capital)
    
    position = 0
    cash = float(initial_capital)
    shares = 0
    trades = []
    
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'sma_20']) or pd.isna(df.loc[i, 'sma_50']):
            df.loc[i, 'position'] = position
            df.loc[i, 'cash'] = cash
            df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
            df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
            continue
        
        # Buy signal: short MA crosses above long MA
        if (df.loc[i-1, 'sma_20'] <= df.loc[i-1, 'sma_50'] and 
            df.loc[i, 'sma_20'] > df.loc[i, 'sma_50'] and 
            position == 0 and cash > 0):
            
            shares = int(cash / df.loc[i, 'close'])
            if shares > 0:
                cost = shares * df.loc[i, 'close']
                cash -= cost
                position = 1
                trades.append({
                    'date': df.loc[i, 'date'],
                    'type': 'BUY',
                    'price': df.loc[i, 'close'],
                    'shares': shares,
                    'value': cost
                })
        
        # Sell signal: short MA crosses below long MA
        elif (df.loc[i-1, 'sma_20'] >= df.loc[i-1, 'sma_50'] and 
              df.loc[i, 'sma_20'] < df.loc[i, 'sma_50'] and 
              position == 1 and shares > 0):
            
            revenue = shares * df.loc[i, 'close']
            cash += revenue
            trades.append({
                'date': df.loc[i, 'date'],
                'type': 'SELL',
                'price': df.loc[i, 'close'],
                'shares': shares,
                'value': revenue
            })
            shares = 0
            position = 0
        
        df.loc[i, 'position'] = position
        df.loc[i, 'cash'] = cash
        df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
        df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
    
    return df, trades

def backtest_macd_strategy(df, initial_capital):
    """MACD crossover strategy"""
    df = df.copy().reset_index(drop=True)
    df['position'] = 0
    df['cash'] = float(initial_capital)
    df['holdings'] = 0.0
    df['total'] = float(initial_capital)
    
    position = 0
    cash = float(initial_capital)
    shares = 0
    trades = []
    
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'macd']) or pd.isna(df.loc[i, 'macd_signal']):
            df.loc[i, 'position'] = position
            df.loc[i, 'cash'] = cash
            df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
            df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
            continue
        
        # Buy signal: MACD crosses above signal
        if (df.loc[i-1, 'macd'] <= df.loc[i-1, 'macd_signal'] and 
            df.loc[i, 'macd'] > df.loc[i, 'macd_signal'] and 
            position == 0 and cash > 0):
            
            shares = int(cash / df.loc[i, 'close'])
            if shares > 0:
                cost = shares * df.loc[i, 'close']
                cash -= cost
                position = 1
                trades.append({
                    'date': df.loc[i, 'date'],
                    'type': 'BUY',
                    'price': df.loc[i, 'close'],
                    'shares': shares,
                    'value': cost
                })
        
        # Sell signal: MACD crosses below signal
        elif (df.loc[i-1, 'macd'] >= df.loc[i-1, 'macd_signal'] and 
              df.loc[i, 'macd'] < df.loc[i, 'macd_signal'] and 
              position == 1 and shares > 0):
            
            revenue = shares * df.loc[i, 'close']
            cash += revenue
            trades.append({
                'date': df.loc[i, 'date'],
                'type': 'SELL',
                'price': df.loc[i, 'close'],
                'shares': shares,
                'value': revenue
            })
            shares = 0
            position = 0
        
        df.loc[i, 'position'] = position
        df.loc[i, 'cash'] = cash
        df.loc[i, 'holdings'] = shares * df.loc[i, 'close']
        df.loc[i, 'total'] = cash + (shares * df.loc[i, 'close'])
    
    return df, trades

# Run Backtest Button
st.sidebar.markdown("---")
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

if run_backtest:
    with st.spinner("Running backtest..."):
        
        # Execute strategy
        if strategy_type == "RSI Strategy":
            result_df, trades = backtest_rsi_strategy(df, rsi_buy, rsi_sell, initial_capital)
        elif strategy_type == "Moving Average Crossover":
            result_df, trades = backtest_ma_crossover(df, ma_short, ma_long, initial_capital)
        elif strategy_type == "MACD Strategy":
            result_df, trades = backtest_macd_strategy(df, initial_capital)
        else:
            st.warning("This strategy is under development")
            st.stop()
        
        # Calculate metrics
        final_value = result_df['total'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Buy & Hold comparison
        buy_hold_shares = initial_capital / df['close'].iloc[0]
        buy_hold_value = buy_hold_shares * df['close'].iloc[-1]
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        # Trade statistics
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        winning_trades = 0
        losing_trades = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                winning_trades += 1
            else:
                losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Store in session state
        st.session_state['backtest_results'] = {
            'result_df': result_df,
            'trades': trades,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_value': buy_hold_value,
            'buy_hold_return': buy_hold_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'initial_capital': initial_capital,
            'strategy': strategy_type,
            'stock': selected_stock
        }

# Display Results
if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    
    st.success(f"✅ Backtest Complete: {results['strategy']} on {results['stock']}")
    
    # Key Metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Final Value",
            f"₹{results['final_value']:,.0f}",
            delta=f"{results['total_return']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Strategy Return",
            f"{results['total_return']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Buy & Hold",
            f"{results['buy_hold_return']:.2f}%"
        )
    
    with col4:
        alpha = results['total_return'] - results['buy_hold_return']
        st.metric(
            "Alpha",
            f"{alpha:.2f}%",
            delta="Outperform" if alpha > 0 else "Underperform"
        )
    
    with col5:
        st.metric(
            "Win Rate",
            f"{results['win_rate']:.1f}%"
        )
    
    # Portfolio Value Chart
    st.markdown("---")
    st.subheader("📈 Portfolio Value Over Time")
    
    result_df = results['result_df']
    trades = results['trades']
    
    fig = go.Figure()
    
    # Strategy performance
    fig.add_trace(go.Scatter(
        x=result_df['date'],
        y=result_df['total'],
        name='Strategy Portfolio',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty'
    ))
    
    # Buy & Hold comparison
    buy_hold_values = [results['initial_capital']] + \
                     [(results['initial_capital'] / df['close'].iloc[0]) * price 
                      for price in df['close'].iloc[1:]]
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=buy_hold_values,
        name='Buy & Hold',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Mark trades
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_values = [result_df[result_df['date'] == d]['total'].values[0] 
                         if len(result_df[result_df['date'] == d]) > 0 else 0
                         for d in buy_dates]
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_values,
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ))
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_values = [result_df[result_df['date'] == d]['total'].values[0]
                          if len(result_df[result_df['date'] == d]) > 0 else 0
                          for d in sell_dates]
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_values,
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=12, symbol='triangle-down')
            ))
    
    fig.update_layout(
        yaxis_title='Portfolio Value (₹)',
        xaxis_title='Date',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade History
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Trade History")
    
    with col2:
        if trades:
            trades_df = pd.DataFrame(trades)
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trades",
                csv,
                f"trades_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        trades_df['price'] = trades_df['price'].round(2)
        trades_df['value'] = trades_df['value'].round(2)
        
        # Color code trades
        def highlight_trades(row):
            if row['type'] == 'BUY':
                return ['background-color: lightgreen'] * len(row)
            else:
                return ['background-color: lightcoral'] * len(row)
        
        styled_trades = trades_df.style.apply(highlight_trades, axis=1)
        st.dataframe(styled_trades, use_container_width=True)
    else:
        st.info("No trades executed during this period")
    
    # Detailed Statistics
    st.markdown("---")
    st.subheader("📊 Detailed Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Returns Analysis**")
        st.write(f"- Initial Capital: ₹{results['initial_capital']:,.2f}")
        st.write(f"- Final Value: ₹{results['final_value']:,.2f}")
        st.write(f"- Absolute Return: ₹{results['final_value'] - results['initial_capital']:,.2f}")
        st.write(f"- Percentage Return: {results['total_return']:.2f}%")
        st.write(f"- Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        st.write(f"- Alpha: {results['total_return'] - results['buy_hold_return']:.2f}%")
    
    with col2:
        st.write("**Trade Statistics**")
        st.write(f"- Total Trades: {results['total_trades']}")
        st.write(f"- Winning Trades: {results['winning_trades']}")
        st.write(f"- Losing Trades: {results['losing_trades']}")
        st.write(f"- Win Rate: {results['win_rate']:.1f}%")
        if results['total_trades'] > 0:
            avg_return_per_trade = results['total_return'] / results['total_trades']
            st.write(f"- Avg Return/Trade: {avg_return_per_trade:.2f}%")
    
    with col3:
        st.write("**Risk Metrics**")
        # Calculate max drawdown
        cumulative = result_df['total']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        st.write(f"- Max Drawdown: {max_drawdown:.2f}%")
        st.write(f"- Final Position: {'IN' if result_df['position'].iloc[-1] == 1 else 'OUT'}")
        st.write(f"- Days in Market: {result_df[result_df['position'] == 1].shape[0]}")
        st.write(f"- Market Exposure: {(result_df[result_df['position'] == 1].shape[0] / len(result_df) * 100):.1f}%")

else:
    # Show strategy information before backtest
    st.info("👆 Configure your strategy in the sidebar and click 'Run Backtest' to begin")
    
    st.markdown("---")
    st.subheader("📖 Strategy Explanations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["RSI Strategy", "MA Crossover", "MACD Strategy", "Bollinger Bands"])
    
    with tab1:
        st.markdown("""
        ### RSI (Relative Strength Index) Strategy
        
        **Buy Signal:** When RSI drops below the buy threshold (default: 30)
        - Indicates the stock is **oversold**
        - Potential bounce back expected
        
        **Sell Signal:** When RSI rises above the sell threshold (default: 70)
        - Indicates the stock is **overbought**
        - Potential pullback expected
        
        **Best For:** Range-bound markets, mean reversion trading
        **Risk:** May generate many false signals in strong trending markets
        """)
    
    with tab2:
        st.markdown("""
        ### Moving Average Crossover Strategy
        
        **Buy Signal:** When short-term MA crosses **above** long-term MA (Golden Cross)
        - Indicates upward momentum
        - Trend reversal to the upside
        
        **Sell Signal:** When short-term MA crosses **below** long-term MA (Death Cross)
        - Indicates downward momentum
        - Trend reversal to the downside
        
        **Best For:** Trending markets, trend-following strategies
        **Risk:** Lagging indicator, may miss early trend reversals
        """)
    
    with tab3:
        st.markdown("""
        ### MACD (Moving Average Convergence Divergence) Strategy
        
        **Buy Signal:** When MACD line crosses **above** signal line
        - Bullish momentum shift
        - Potential uptrend beginning
        
        **Sell Signal:** When MACD line crosses **below** signal line
        - Bearish momentum shift
        - Potential downtrend beginning
        
        **Best For:** Markets with clear momentum, swing trading
        **Risk:** Whipsaws in sideways markets
        """)
    
    with tab4:
        st.markdown("""
        ### Bollinger Bands Strategy
        
        **Buy Signal:** When price touches or falls below lower band
        - Stock oversold relative to recent volatility
        - Mean reversion expected
        
        **Sell Signal:** When price touches or rises above upper band
        - Stock overbought relative to recent volatility
        - Pullback to mean expected
        
        **Best For:** Volatile markets, range-bound trading
        **Risk:** Bands can expand during strong trends, causing early exits
        """)

st.markdown("---")
st.caption("⚠️ **Disclaimer:** Past performance does not guarantee future results. Use backtesting for educational purposes only.")