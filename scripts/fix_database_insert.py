import sys
sys.path.append('.')

from src.loaders.database import DatabaseManager
from src.extractors.yahoo_finance import YahooFinanceExtractor
from src.transformers.technical_indicators import TechnicalIndicators
from src.config.settings import settings
import pandas as pd
from datetime import datetime

print("🔧 Fixing database insertion issue...\n")

# Initialize
db = DatabaseManager()
extractor = YahooFinanceExtractor()
calculator = TechnicalIndicators()

# Test with one stock first
test_symbol = "RELIANCE.NS"
print(f"Testing with {test_symbol}...")

# Fetch data
df = extractor.get_stock_data(test_symbol, period="1mo")

if df is not None and not df.empty:
    print(f"✓ Fetched {len(df)} records")
    
    # Clean the data
    df_clean = df.copy()
    
    # Remove timezone from date
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
    
    # Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean['volume'] = pd.to_numeric(df_clean['volume'], errors='coerce').fillna(0).astype('int64')
    
    # Remove NaN
    df_clean = df_clean.dropna()
    
    print(f"✓ Cleaned data: {len(df_clean)} records")
    
    # Insert row by row
    conn = db.engine.raw_connection()
    cursor = conn.cursor()
    
    success_count = 0
    for _, row in df_clean.iterrows():
        try:
            cursor.execute("""
                INSERT INTO stock_prices (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume
            """, (
                row['symbol'],
                row['date'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['adj_close']),
                int(row['volume'])
            ))
            success_count += 1
        except Exception as e:
            print(f"✗ Error inserting row: {e}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✓ Successfully inserted {success_count} records!")
    
    # Verify
    import psycopg2
    check_conn = psycopg2.connect(db.connection_string)
    check_cursor = check_conn.cursor()
    check_cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE symbol = %s", (test_symbol,))
    count = check_cursor.fetchone()[0]
    check_cursor.close()
    check_conn.close()
    
    print(f"\n✅ Verification: {count} records in database for {test_symbol}")
    
    if count > 0:
        print("\n🎉 Success! Database insertion is now working!")
        print("\nNow run the full data load:")
        print("python scripts/initial_data_load.py")
else:
    print("✗ Failed to fetch data")