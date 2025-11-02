import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.loaders.database import DatabaseManager
from src.config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Stock Market Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()

# Sidebar
st.sidebar.title("📊 Stock Market Analytics")
st.sidebar.markdown("---")

# Stock selection
available_stocks = settings.NIFTY_50_STOCKS
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    options=available_stocks,
    index=0
)

# Date range selection
date_range = st.sidebar.selectbox(
    "Date Range",
    options=["1m","5m","15m","1h","4h","1d","1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
    index=2
)

# Map date range to days
range_mapping = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "240m",
    "1d": "1d",
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}
limit_mapping = {
    "1m": 500,
    "5m": 500,
    "15m": 500,
    "1h": 500,
    "4h": 500,
    "1d": 500,
    "1 Week": 700,
    "1 Month": 3000,
    "3 Months": 9000,
    "6 Months": 18000,
    "1 Year": 36500
}
days = range_mapping[date_range]
selected_limit = limit_mapping[date_range]


st.sidebar.markdown("---")
st.sidebar.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main content
st.title(f"📈 {selected_stock} - Stock Analysis Dashboard")

# Fetch data
@st.cache_data(ttl=300)
def load_stock_data(symbol, days):
    price_df = db.get_latest_stock_data(symbol, limit=selected_limit)
    indicators_df = db.get_latest_indicators(symbol, limit=selected_limit)
    
    if price_df.empty or indicators_df.empty:
        return pd.DataFrame()
    
    merged_df = price_df.merge(indicators_df, on=['symbol', 'date'], how='inner')
    merged_df = merged_df.sort_values('date')
    return merged_df

df = load_stock_data(selected_stock, days)

if df.empty:
    st.error(f"No data available for {selected_stock}. Please run the data pipeline first.")
    st.stop()

# Key Metrics Row
latest_data = df.iloc[-1]
previous_data = df.iloc[-2] if len(df) > 1 else latest_data

col1, col2, col3, col4 = st.columns(4)

with col1:
    price_change = latest_data['close'] - previous_data['close']
    price_change_pct = (price_change / previous_data['close']) * 100
    st.metric(
        label="Current Price",
        value=f"₹{latest_data['close']:.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col2:
    st.metric(
        label="Day High",
        value=f"₹{latest_data['high']:.2f}"
    )

with col3:
    st.metric(
        label="Day Low",
        value=f"₹{latest_data['low']:.2f}"
    )

with col4:
    volume_millions = latest_data['volume'] / 1_000_000
    st.metric(
        label="Volume",
        value=f"{volume_millions:.2f}M"
    )

# RSI and MACD metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    rsi_value = latest_data.get('rsi_14', 0)
    rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
    st.metric(label="RSI (14)", value=f"{rsi_value:.2f}", delta=rsi_status)

with col2:
    macd_value = latest_data.get('macd', 0)
    st.metric(label="MACD", value=f"{macd_value:.4f}")

with col3:
    sma_20 = latest_data.get('sma_20', 0)
    st.metric(label="SMA (20)", value=f"₹{sma_20:.2f}")

with col4:
    sma_50 = latest_data.get('sma_50', 0)
    st.metric(label="SMA (50)", value=f"₹{sma_50:.2f}")

st.markdown("---")

# Price Chart with Moving Averages
st.subheader("📊 Price Chart with Technical Indicators")

fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='Price'
))

# Add Moving Averages
if 'sma_20' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))

if 'sma_50' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_50'],
        name='SMA 50',
        line=dict(color='blue', width=1)
    ))

if 'sma_200' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_200'],
        name='SMA 200',
        line=dict(color='red', width=1)
    ))

fig.update_layout(
    title=f'{selected_stock} Price Chart',
    yaxis_title='Price (₹)',
    xaxis_title='Date',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Two columns for additional charts
col1, col2 = st.columns(2)

with col1:
    # RSI Chart
    st.subheader("📉 RSI Indicator")
    
    fig_rsi = go.Figure()
    
    if 'rsi_14' in df.columns:
        fig_rsi.add_trace(go.Scatter(
            x=df['date'],
            y=df['rsi_14'],
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(
            yaxis_title='RSI',
            xaxis_title='Date',
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    # MACD Chart
    st.subheader("📊 MACD Indicator")
    
    fig_macd = go.Figure()
    
    if 'macd' in df.columns:
        fig_macd.add_trace(go.Scatter(
            x=df['date'],
            y=df['macd'],
            name='MACD',
            line=dict(color='blue', width=2)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=df['date'],
            y=df['macd_signal'],
            name='Signal',
            line=dict(color='orange', width=2)
        ))
        
        # Add histogram
        colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
        fig_macd.add_trace(go.Bar(
            x=df['date'],
            y=df['macd_histogram'],
            name='Histogram',
            marker_color=colors
        ))
        
        fig_macd.update_layout(
            yaxis_title='MACD',
            xaxis_title='Date',
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)

# Volume Chart
st.subheader("📊 Trading Volume")
fig_volume = go.Figure()

fig_volume.add_trace(go.Bar(
    x=df['date'],
    y=df['volume'],
    name='Volume',
    marker_color='lightblue'
))

fig_volume.update_layout(
    yaxis_title='Volume',
    xaxis_title='Date',
    height=300,
    hovermode='x unified'
)

st.plotly_chart(fig_volume, use_container_width=True)

# Recent Alerts
st.markdown("---")
st.subheader("🚨 Recent Alerts")

@st.cache_data(ttl=60)
def load_recent_alerts(symbol):
    query = f"""
        SELECT alert_type, condition_met, value, triggered_at
        FROM alerts
        WHERE symbol = '{symbol}'
        ORDER BY triggered_at DESC
        LIMIT 10
    """
    return pd.read_sql(query, db.engine)

alerts_df = load_recent_alerts(selected_stock)

if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.info("No recent alerts for this stock.")

# Data Table
st.markdown("---")
st.subheader("📋 Recent Data")

display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 
                   'rsi_14', 'macd', 'sma_20', 'sma_50']
available_display_cols = [col for col in display_columns if col in df.columns]

st.dataframe(
    df[available_display_cols].tail(20).sort_values('date', ascending=False),
    use_container_width=True
)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name=f"{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)