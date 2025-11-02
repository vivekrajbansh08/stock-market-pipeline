import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
sys.path.append('../..')

from src.loaders.database import DatabaseManager
from src.models.price_predictor import StockPricePredictor
from src.config.settings import settings

st.set_page_config(page_title="AI Price Predictions", page_icon="🤖", layout="wide")

# Initialize
@st.cache_resource
def get_services():
    return DatabaseManager(), StockPricePredictor()

db, predictor = get_services()

st.title("🤖 AI-Powered Price Predictions")
st.markdown("Machine Learning models to predict future stock prices")

# Sidebar configuration
st.sidebar.header("⚙️ Model Configuration")

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    settings.NIFTY_50_STOCKS
)

prediction_days = st.sidebar.slider(
    "Prediction Horizon (days)",
    min_value=1,
    max_value=30,
    value=5
)

train_data_period = st.sidebar.selectbox(
    "Training Data Period",
    ["3 Months", "6 Months", "1 Year"],
    index=2
)

period_mapping = {"3 Months": 90, "6 Months": 180, "1 Year": 365}
days = period_mapping[train_data_period]

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Random Forest", "LSTM (Advanced)"]
)

# Load data
@st.cache_data(ttl=300)
def load_data(symbol, days):
    price_df = db.get_latest_stock_data(symbol, days)
    indicators_df = db.get_latest_indicators(symbol, days)
    
    if price_df.empty:
        return pd.DataFrame()
    
    if not indicators_df.empty:
        merged_df = price_df.merge(indicators_df, on=['symbol', 'date'], how='left')
    else:
        merged_df = price_df
    
    return merged_df.sort_values('date').reset_index(drop=True)

df = load_data(selected_stock, days)

if df.empty:
    st.error(f"❌ No data available for {selected_stock}. Please run the data pipeline first.")
    st.info("""
    **To load data:**
    ```bash
    python scripts/initial_data_load.py
    ```
    """)
    st.stop()

# Train and predict button
st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Train Model & Predict", type="primary", use_container_width=True)

# Current price info
st.subheader(f"📊 {selected_stock} Overview")

latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest

col1, col2, col3, col4 = st.columns(4)

with col1:
    price_change = latest['close'] - previous['close']
    price_change_pct = (price_change / previous['close']) * 100
    st.metric(
        "Current Price",
        f"₹{latest['close']:.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col2:
    st.metric("High", f"₹{latest['high']:.2f}")

with col3:
    st.metric("Low", f"₹{latest['low']:.2f}")

with col4:
    volume_m = latest['volume'] / 1_000_000
    st.metric("Volume", f"{volume_m:.2f}M")

# Train model and make predictions
if train_button:
    with st.spinner(f"🤖 Training {model_type} model on {len(df)} days of data..."):
        try:
            # Train model
            if model_type == "Random Forest":
                metrics = predictor.train_random_forest(df, test_size=0.2)
                
                # Make predictions
                predictions = predictor.predict_next_days(df, days=prediction_days)
                
                # Store in session state
                st.session_state['predictions'] = predictions
                st.session_state['metrics'] = metrics
                st.session_state['model_type'] = model_type
                st.session_state['trained_stock'] = selected_stock
                
                st.success(f"✅ Model trained successfully! Test R² Score: {metrics['test_r2']:.4f}")
                
            else:  # LSTM
                try:
                    from src.models.price_predictor import LSTMStockPredictor
                    lstm_predictor = LSTMStockPredictor(sequence_length=60)
                    metrics = lstm_predictor.train_lstm(df, epochs=50, batch_size=32)
                    predictions = lstm_predictor.predict_next_days(df, days=prediction_days)
                    
                    st.session_state['predictions'] = predictions
                    st.session_state['metrics'] = metrics
                    st.session_state['model_type'] = model_type
                    st.session_state['trained_stock'] = selected_stock
                    
                    st.success(f"✅ LSTM model trained! Test RMSE: {metrics['test_rmse']:.2f}")
                    
                except ImportError:
                    st.error("❌ TensorFlow not installed. Install with: `pip install tensorflow`")
                    st.info("Using Random Forest instead...")
                    metrics = predictor.train_random_forest(df, test_size=0.2)
                    predictions = predictor.predict_next_days(df, days=prediction_days)
                    
                    st.session_state['predictions'] = predictions
                    st.session_state['metrics'] = metrics
                    st.session_state['model_type'] = "Random Forest"
                    st.session_state['trained_stock'] = selected_stock
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.stop()

# Display predictions
if 'predictions' in st.session_state and st.session_state.get('trained_stock') == selected_stock:
    predictions = st.session_state['predictions']
    metrics = st.session_state['metrics']
    model_type = st.session_state['model_type']
    
    st.markdown("---")
    st.subheader(f"🔮 Price Predictions ({model_type})")
    
    # Prediction metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        predicted_price = predictions['predicted_price'].iloc[-1]
        current_price = df['close'].iloc[-1]
        price_diff = predicted_price - current_price
        price_diff_pct = (price_diff / current_price) * 100
        
        st.metric(
            f"Predicted Price (Day {prediction_days})",
            f"₹{predicted_price:.2f}",
            delta=f"{price_diff_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            "Potential Gain/Loss",
            f"₹{price_diff:.2f}",
            delta=f"{price_diff_pct:.2f}%"
        )
    
    with col3:
        st.metric(
            "Model Accuracy (R²)",
            f"{metrics.get('test_r2', metrics.get('test_rmse', 0)):.4f}"
        )
    
    with col4:
        st.metric(
            "Model Error (RMSE)",
            f"₹{metrics.get('test_rmse', 0):.2f}"
        )
    
    # Prediction visualization
    st.markdown("---")
    st.subheader("📈 Price Forecast Chart")
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        mode='lines'
    ))
    
    # Predicted prices
    fig.add_trace(go.Scatter(
        x=predictions['date'],
        y=predictions['predicted_price'],
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Add confidence interval (±5%)
    upper_bound = predictions['predicted_price'] * 1.05
    lower_bound = predictions['predicted_price'] * 0.95
    
    fig.add_trace(go.Scatter(
        x=predictions['date'],
        y=upper_bound,
        name='Upper Bound (+5%)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions['date'],
        y=lower_bound,
        name='Lower Bound (-5%)',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{selected_stock} Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction table
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Daily Predictions")
    
    with col2:
        csv = predictions.to_csv(index=False)
        st.download_button(
            "📥 Download",
            csv,
            f"predictions_{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    # Format predictions table
    display_pred = predictions.copy()
    display_pred['date'] = pd.to_datetime(display_pred['date']).dt.strftime('%Y-%m-%d')
    display_pred['predicted_price'] = display_pred['predicted_price'].round(2)
    display_pred['change_from_today'] = ((display_pred['predicted_price'] - current_price) / current_price * 100).round(2)
    
    st.dataframe(
        display_pred[['day', 'date', 'predicted_price', 'change_from_today']],
        use_container_width=True,
        column_config={
            "day": "Day",
            "date": "Date",
            "predicted_price": st.column_config.NumberColumn("Predicted Price (₹)", format="₹%.2f"),
            "change_from_today": st.column_config.NumberColumn("Change (%)", format="%.2f%%")
        }
    )
    
    # Model performance
    st.markdown("---")
    st.subheader("🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Metrics**")
        st.write(f"- RMSE (Root Mean Squared Error): ₹{metrics.get('train_rmse', 0):.2f}")
        st.write(f"- MAE (Mean Absolute Error): ₹{metrics.get('train_mae', 0):.2f}")
        st.write(f"- R² Score: {metrics.get('train_r2', 0):.4f}")
        st.write(f"- MAPE: {metrics.get('train_mape', 0):.2f}%")
    
    with col2:
        st.write("**Testing Metrics**")
        st.write(f"- RMSE (Root Mean Squared Error): ₹{metrics.get('test_rmse', 0):.2f}")
        st.write(f"- MAE (Mean Absolute Error): ₹{metrics.get('test_mae', 0):.2f}")
        st.write(f"- R² Score: {metrics.get('test_r2', 0):.4f}")
        st.write(f"- MAPE: {metrics.get('test_mape', 0):.2f}%")
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        st.markdown("---")
        st.subheader("🔍 Feature Importance")
        
        importance_df = predictor.get_feature_importance()
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'].head(10),
            y=importance_df['feature'].head(10),
            orientation='h',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title='Top 10 Most Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Investment recommendation
    st.markdown("---")
    st.subheader("💡 AI Recommendation")
    
    predicted_change = (predicted_price - current_price) / current_price * 100
    
    if predicted_change > 3:
        st.success(f"""
        **🟢 BUY SIGNAL**
        
        The model predicts a **{predicted_change:.2f}%** increase over the next {prediction_days} days.
        
        - Current Price: ₹{current_price:.2f}
        - Predicted Price: ₹{predicted_price:.2f}
        - Potential Gain: ₹{price_diff:.2f} ({predicted_change:.2f}%)
        
        **Risk Level:** Medium
        """)
    elif predicted_change < -3:
        st.error(f"""
        **🔴 SELL SIGNAL**
        
        The model predicts a **{predicted_change:.2f}%** decrease over the next {prediction_days} days.
        
        - Current Price: ₹{current_price:.2f}
        - Predicted Price: ₹{predicted_price:.2f}
        - Potential Loss: ₹{price_diff:.2f} ({predicted_change:.2f}%)
        
        **Risk Level:** Medium-High
        """)
    else:
        st.info(f"""
        **🟡 HOLD**
        
        The model predicts a **{predicted_change:.2f}%** change over the next {prediction_days} days.
        
        - Current Price: ₹{current_price:.2f}
        - Predicted Price: ₹{predicted_price:.2f}
        - Expected Change: ₹{price_diff:.2f} ({predicted_change:.2f}%)
        
        **Risk Level:** Low
        """)

else:
    # Show instructions before training
    st.info("👆 Configure your model in the sidebar and click 'Train Model & Predict' to generate predictions")
    
    st.markdown("---")
    st.subheader("🤖 About AI Price Prediction")
    
    tab1, tab2, tab3 = st.tabs(["How it Works", "Model Types", "Accuracy"])
    
    with tab1:
        st.markdown("""
        ### How AI Prediction Works
        
        1. **Data Collection**: Historical stock prices and technical indicators
        2. **Feature Engineering**: Create predictive features from historical data
            - Price lags (previous days' prices)
            - Rolling averages and standard deviations
            - Technical indicators (RSI, MACD, Moving Averages)
        3. **Model Training**: Train ML model on historical patterns
        4. **Prediction**: Generate future price predictions
        5. **Confidence Intervals**: Calculate uncertainty ranges
        
        The model learns patterns from past data to predict future movements.
        """)
    
    with tab2:
        st.markdown("""
        ### Available Models
        
        **1. Random Forest (Default)**
        - **Pros**: Fast, accurate, explainable
        - **Cons**: May miss complex patterns
        - **Best For**: Short-term predictions (1-7 days)
        - **Training Time**: ~10 seconds
        
        **2. LSTM (Advanced)**
        - **Pros**: Captures complex temporal patterns
        - **Cons**: Slower, requires more data, harder to interpret
        - **Best For**: Medium-term predictions (7-30 days)
        - **Training Time**: ~2-5 minutes
        - **Requires**: TensorFlow (`pip install tensorflow`)
        """)
    
    with tab3:
        st.markdown("""
        ### Understanding Accuracy Metrics
        
        **R² Score (R-Squared)**
        - Range: 0 to 1 (higher is better)
        - 0.8+ = Excellent
        - 0.6-0.8 = Good
        - 0.4-0.6 = Moderate
        - <0.4 = Poor
        
        **RMSE (Root Mean Squared Error)**
        - Lower is better
        - Shows average prediction error in rupees
        - Example: RMSE of ₹10 means predictions are off by ~₹10 on average
        
        **MAPE (Mean Absolute Percentage Error)**
        - Shows average error as percentage
        - <5% = Excellent
        - 5-10% = Good
        - 10-20% = Moderate
        - >20% = Poor
        """)

# Disclaimer
st.markdown("---")
st.warning("""
⚠️ **Disclaimer**: 
- AI predictions are based on historical patterns and may not reflect future performance
- Use predictions as one of many factors in investment decisions
- Past performance does not guarantee future results
- Always do your own research and consult financial advisors
- This tool is for educational purposes only
""")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")