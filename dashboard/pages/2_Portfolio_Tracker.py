import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
sys.path.append('../..')

from src.loaders.database import DatabaseManager
from src.extractors.yahoo_finance import YahooFinanceExtractor
from src.config.settings import settings

st.set_page_config(page_title="Portfolio Tracker", page_icon="💼", layout="wide")

# Initialize
@st.cache_resource
def get_services():
    return DatabaseManager(), YahooFinanceExtractor()

db, extractor = get_services()

st.title("💼 Portfolio Tracker")
st.markdown("Track your stock portfolio performance in real-time")

# Check if portfolio table exists, if not create it
try:
    query = "SELECT * FROM portfolio LIMIT 1"
    pd.read_sql(query, db.engine)
except:
    st.warning("Portfolio table not found. Creating it now...")
    create_table = """
        CREATE TABLE IF NOT EXISTS portfolio (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            quantity DECIMAL(15, 4) NOT NULL,
            avg_buy_price DECIMAL(15, 4) NOT NULL,
            purchase_date DATE NOT NULL,
            current_price DECIMAL(15, 4),
            current_value DECIMAL(15, 4),
            profit_loss DECIMAL(15, 4),
            profit_loss_pct DECIMAL(10, 4),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol)
        );
    """
    db.engine.execute(create_table)
    st.success("Portfolio table created!")

# Sidebar - Add Stock
st.sidebar.header("➕ Add to Portfolio")

with st.sidebar.form("add_stock_form"):
    symbol = st.selectbox("Stock Symbol", settings.NIFTY_50_STOCKS)
    quantity = st.number_input("Quantity", min_value=1, value=10)
    buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=1000.0, step=10.0)
    purchase_date = st.date_input("Purchase Date")
    
    submitted = st.form_submit_button("Add to Portfolio")
    
    if submitted:
        try:
            # Get current price
            latest_info = extractor.get_latest_price(symbol)
            current_price = latest_info['current_price'] if latest_info else buy_price
            
            # Calculate metrics
            current_value = quantity * current_price
            investment = quantity * buy_price
            profit_loss = current_value - investment
            profit_loss_pct = (profit_loss / investment) * 100
            
            # Check if stock already exists
            query = f"SELECT * FROM portfolio WHERE symbol = '{symbol}'"
            existing = pd.read_sql(query, db.engine)
            
            if not existing.empty:
                # Update existing entry - calculate new average
                old_qty = float(existing['quantity'].iloc[0])
                old_avg = float(existing['avg_buy_price'].iloc[0])
                new_qty = old_qty + quantity
                new_avg = ((old_avg * old_qty) + (buy_price * quantity)) / new_qty
                
                update_query = f"""
                    UPDATE portfolio 
                    SET quantity = {new_qty},
                        avg_buy_price = {new_avg},
                        current_price = {current_price},
                        current_value = {new_qty * current_price},
                        profit_loss = {(new_qty * current_price) - (new_qty * new_avg)},
                        profit_loss_pct = {(((new_qty * current_price) - (new_qty * new_avg)) / (new_qty * new_avg)) * 100},
                        last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = '{symbol}'
                """
                db.engine.execute(update_query)
                st.sidebar.success(f"Updated {symbol}! New quantity: {new_qty}")
            else:
                # Insert new entry
                portfolio_data = pd.DataFrame([{
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_buy_price': buy_price,
                    'purchase_date': purchase_date,
                    'current_price': current_price,
                    'current_value': current_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct
                }])
                portfolio_data.to_sql('portfolio', db.engine, if_exists='append', index=False)
                st.sidebar.success(f"Added {quantity} shares of {symbol}!")
            
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Load portfolio data
@st.cache_data(ttl=60)
def load_portfolio():
    query = "SELECT * FROM portfolio ORDER BY symbol"
    try:
        return pd.read_sql(query, db.engine)
    except:
        return pd.DataFrame()

portfolio_df = load_portfolio()

if portfolio_df.empty:
    st.info("📝 Your portfolio is empty. Add stocks using the sidebar!")
    st.markdown("---")
    st.subheader("📚 How to use Portfolio Tracker")
    st.markdown("""
    1. **Select a stock** from the dropdown in the sidebar
    2. **Enter quantity** and purchase price
    3. **Select purchase date**
    4. **Click 'Add to Portfolio'**
    5. View your portfolio performance below
    6. Use **'Update Prices'** button to refresh current prices
    """)
    st.stop()

# Update current prices button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Update All Prices", type="primary"):
    with st.spinner("Updating prices..."):
        updated_count = 0
        for idx, row in portfolio_df.iterrows():
            try:
                latest_info = extractor.get_latest_price(row['symbol'])
                if latest_info and latest_info['current_price']:
                    current_price = latest_info['current_price']
                    current_value = float(row['quantity']) * current_price
                    investment = float(row['quantity']) * float(row['avg_buy_price'])
                    profit_loss = current_value - investment
                    profit_loss_pct = (profit_loss / investment) * 100
                    
                    update_query = f"""
                        UPDATE portfolio 
                        SET current_price = {current_price},
                            current_value = {current_value},
                            profit_loss = {profit_loss},
                            profit_loss_pct = {profit_loss_pct},
                            last_updated = CURRENT_TIMESTAMP
                        WHERE symbol = '{row['symbol']}'
                    """
                    db.engine.execute(update_query)
                    updated_count += 1
            except Exception as e:
                st.sidebar.warning(f"Failed to update {row['symbol']}: {str(e)}")
                continue
        
        st.sidebar.success(f"Updated {updated_count}/{len(portfolio_df)} stocks!")
        st.rerun()

# Portfolio Summary
st.subheader("📊 Portfolio Summary")

total_investment = (portfolio_df['quantity'] * portfolio_df['avg_buy_price']).sum()
total_current_value = portfolio_df['current_value'].sum()
total_profit_loss = total_current_value - total_investment
total_profit_loss_pct = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Investment", f"₹{total_investment:,.2f}")

with col2:
    st.metric("Current Value", f"₹{total_current_value:,.2f}")

with col3:
    st.metric(
        "Total P&L", 
        f"₹{total_profit_loss:,.2f}",
        delta=f"{total_profit_loss_pct:.2f}%",
        delta_color="normal"
    )

with col4:
    winning_stocks = len(portfolio_df[portfolio_df['profit_loss'] > 0])
    total_stocks = len(portfolio_df)
    st.metric("Winning/Total", f"{winning_stocks}/{total_stocks}")

# Portfolio Composition
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Portfolio Allocation")
    
    fig = px.pie(
        portfolio_df,
        values='current_value',
        names='symbol',
        title='Holdings by Current Value',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Profit/Loss by Stock")
    
    # Sort by profit/loss
    sorted_df = portfolio_df.sort_values('profit_loss', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in sorted_df['profit_loss']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_df['symbol'],
            x=sorted_df['profit_loss'],
            orientation='h',
            marker_color=colors,
            text=sorted_df['profit_loss_pct'].round(2),
            texttemplate='%{text}%',
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='Profit/Loss (₹)',
        xaxis_title='Amount (₹)',
        yaxis_title='Stock',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Detailed Holdings Table
st.markdown("---")
st.subheader("📋 Detailed Holdings")

# Format the dataframe for display
display_df = portfolio_df.copy()
display_df['investment'] = (display_df['quantity'] * display_df['avg_buy_price']).round(2)
display_df['avg_buy_price'] = display_df['avg_buy_price'].round(2)
display_df['current_price'] = display_df['current_price'].round(2)
display_df['current_value'] = display_df['current_value'].round(2)
display_df['profit_loss'] = display_df['profit_loss'].round(2)
display_df['profit_loss_pct'] = display_df['profit_loss_pct'].round(2)

# Select columns to display
display_cols = ['symbol', 'quantity', 'avg_buy_price', 'current_price', 
                'investment', 'current_value', 'profit_loss', 'profit_loss_pct', 'purchase_date']

# Color code the table
def color_pnl(val):
    if isinstance(val, (int, float)):
        color = 'lightgreen' if val > 0 else 'lightcoral' if val < 0 else 'white'
        return f'background-color: {color}'
    return ''

styled_df = display_df[display_cols].style.applymap(
    color_pnl, 
    subset=['profit_loss', 'profit_loss_pct']
)

st.dataframe(styled_df, use_container_width=True, height=400)

# Performance Metrics
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top Performers")
    if not portfolio_df.empty:
        top_performers = portfolio_df.nlargest(5, 'profit_loss_pct')[['symbol', 'profit_loss_pct', 'profit_loss']]
        for idx, row in top_performers.iterrows():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.success(f"**{row['symbol']}**")
            with col_b:
                st.metric("", f"+{row['profit_loss_pct']:.2f}%", f"₹{row['profit_loss']:,.2f}")

with col2:
    st.subheader("📉 Bottom Performers")
    if not portfolio_df.empty:
        bottom_performers = portfolio_df.nsmallest(5, 'profit_loss_pct')[['symbol', 'profit_loss_pct', 'profit_loss']]
        for idx, row in bottom_performers.iterrows():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.error(f"**{row['symbol']}**")
            with col_b:
                st.metric("", f"{row['profit_loss_pct']:.2f}%", f"₹{row['profit_loss']:,.2f}")

# Remove Stock Section
st.markdown("---")
st.subheader("🗑️ Manage Portfolio")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    stock_to_remove = st.selectbox("Select stock to remove", portfolio_df['symbol'].tolist())

with col2:
    st.write("")
    st.write("")

with col3:
    st.write("")
    st.write("")
    if st.button("Remove", type="secondary"):
        try:
            delete_query = f"DELETE FROM portfolio WHERE symbol = '{stock_to_remove}'"
            db.engine.execute(delete_query)
            st.success(f"Removed {stock_to_remove}!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Export Portfolio
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    csv = portfolio_df.to_csv(index=False)
    st.download_button(
        label="📥 Export as CSV",
        data=csv,
        file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    # Clear all portfolio
    if st.button("🗑️ Clear All Portfolio", type="secondary", use_container_width=True):
        if st.session_state.get('confirm_clear'):
            try:
                db.engine.execute("DELETE FROM portfolio")
                st.success("Portfolio cleared!")
                st.session_state['confirm_clear'] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.session_state['confirm_clear'] = True
            st.warning("⚠️ Click again to confirm")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")