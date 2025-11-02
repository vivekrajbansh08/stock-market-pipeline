import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('../..')

from src.loaders.database import DatabaseManager
from src.config.settings import settings

st.set_page_config(page_title="Technical Analysis", page_icon="📊", layout="wide")

# Initialize database
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()

st.title("📊 Advanced Technical Analysis")

# Sidebar
st.sidebar.header("Configuration")
symbols = settings.NIFTY_50_STOCKS
selected_stock = st.sidebar.selectbox("Select Stock", symbols)

date_range = st.sidebar.selectbox(
    "Time Period",
    ["1 Month", "3 Months", "6 Months", "1 Year"],
    index=2
)

range_mapping = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}
days = range_mapping[date_range]

# Fetch data
@st.cache_data(ttl=300)
def load_data(symbol, days):
    price_df = db.get_latest_stock_data(symbol, days)
    indicators_df = db.get_latest_indicators(symbol, days)
    
    if price_df.empty or indicators_df.empty:
        return pd.DataFrame()
    
    merged_df = price_df.merge(indicators_df, on=['symbol', 'date'], how='inner')
    return merged_df.sort_values('date')

df = load_data(selected_stock, days)

if df.empty:
    st.error(f"No data available for {selected_stock}")
    st.stop()

# Key Metrics
latest = df.iloc[-1]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Current Price", f"₹{latest['close']:.2f}")

with col2:
    rsi = latest.get('rsi_14', 0)
    rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
    st.metric(f"RSI {rsi_color}", f"{rsi:.2f}")

with col3:
    macd = latest.get('macd', 0)
    st.metric("MACD", f"{macd:.4f}")

with col4:
    sma_20 = latest.get('sma_20', 0)
    st.metric("SMA(20)", f"₹{sma_20:.2f}")

with col5:
    volume_m = latest['volume'] / 1_000_000
    st.metric("Volume", f"{volume_m:.2f}M")

st.markdown("---")

# Price Chart with Multiple Indicators
st.subheader("📈 Price Action with Technical Overlays")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.15, 0.15, 0.2],
    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume')
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ),
    row=1, col=1
)

# Moving Averages
colors = {'sma_20': 'orange', 'sma_50': 'blue', 'sma_200': 'red', 'ema_20': 'purple'}
for ma, color in colors.items():
    if ma in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[ma], name=ma.upper(), 
                      line=dict(color=color, width=1)),
            row=1, col=1
        )

# Bollinger Bands
if 'bb_upper' in df.columns:
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['bb_upper'], name='BB Upper',
                  line=dict(color='gray', width=1, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['bb_lower'], name='BB Lower',
                  line=dict(color='gray', width=1, dash='dot'),
                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )

# RSI
if 'rsi_14' in df.columns:
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['rsi_14'], name='RSI',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
if 'macd' in df.columns:
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['macd'], name='MACD',
                  line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal',
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    
    # Histogram
    colors_hist = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['macd_histogram'], name='Histogram',
              marker_color=colors_hist),
        row=3, col=1
    )

# Volume
fig.add_trace(
    go.Bar(x=df['date'], y=df['volume'], name='Volume',
          marker_color='lightblue'),
    row=4, col=1
)

fig.update_layout(
    height=1000,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Technical Signals
st.markdown("---")
st.subheader("🎯 Trading Signals")

col1, col2 = st.columns(2)

with col1:
    st.write("**Trend Signals**")
    
    # Golden/Death Cross
    if 'sma_50' in df.columns and 'sma_200' in df.columns:
        if latest['sma_50'] > latest['sma_200']:
            st.success("✅ Golden Cross - Bullish trend")
        else:
            st.error("❌ Death Cross - Bearish trend")
    
    # Price vs MA
    if latest['close'] > latest.get('sma_20', 0):
        st.info("📈 Price above SMA(20) - Short-term bullish")
    else:
        st.warning("📉 Price below SMA(20) - Short-term bearish")

with col2:
    st.write("**Momentum Signals**")
    
    # RSI
    rsi = latest.get('rsi_14', 50)
    if rsi < 30:
        st.success("🟢 RSI Oversold - Potential buy")
    elif rsi > 70:
        st.error("🔴 RSI Overbought - Potential sell")
    else:
        st.info("🟡 RSI Neutral")
    
    # MACD
    if latest.get('macd', 0) > latest.get('macd_signal', 0):
        st.success("✅ MACD Bullish crossover")
    else:
        st.warning("⚠️ MACD Bearish crossover")

# Support & Resistance Levels
st.markdown("---")
st.subheader("📍 Support & Resistance Levels")

col1, col2, col3 = st.columns(3)

with col1:
    resistance_1 = df['high'].tail(20).max()
    st.metric("Resistance 1", f"₹{resistance_1:.2f}")

with col2:
    current = latest['close']
    st.metric("Current", f"₹{current:.2f}")

with col3:
    support_1 = df['low'].tail(20).min()
    st.metric("Support 1", f"₹{support_1:.2f}")

# Statistics
st.markdown("---")
st.subheader("📊 Statistical Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    volatility = df['close'].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.2f}%")

with col2:
    avg_volume = df['volume'].mean() / 1_000_000
    st.metric("Avg Volume", f"{avg_volume:.2f}M")

with col3:
    price_change = ((latest['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
    st.metric("Period Return", f"{price_change:.2f}%")

with col4:
    high_52w = df['high'].max()
    low_52w = df['low'].min()
    st.metric("52W High", f"₹{high_52w:.2f}")
    st.metric("52W Low", f"₹{low_52w:.2f}")

# Raw Data
with st.expander("📋 View Raw Data"):
    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                    'rsi_14', 'macd', 'sma_20', 'sma_50']
    available_cols = [col for col in display_cols if col in df.columns]
    st.dataframe(df[available_cols].tail(50), use_container_width=True)