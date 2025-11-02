import sys
sys.path.append('.')

from src.extractors.yahoo_finance import YahooFinanceExtractor
from src.transformers.technical_indicators import TechnicalIndicators
from src.loaders.database import DatabaseManager
from src.alerts.alert_manager import AlertManager

symbol = "RELIANCE.NS"
print(f"Testing pipeline for {symbol}...")

extractor = YahooFinanceExtractor()
df = extractor.get_stock_data(symbol, period="1mo")
print(f"✓ Extracted {len(df)} records")

calculator = TechnicalIndicators()
df_indicators = calculator.calculate_all_indicators(df)
print(f"✓ Calculated indicators")

db = DatabaseManager()
db.insert_stock_prices(df)
prepared_indicators = calculator.prepare_for_database(df_indicators)
db.insert_technical_indicators(prepared_indicators)
print(f"✓ Loaded to database")

alert_manager = AlertManager()
merged_df = df.merge(prepared_indicators, on=['symbol', 'date'], how='inner')
alerts = alert_manager.check_all_alerts(merged_df)
print(f"✓ Found {len(alerts)} alerts")

if alerts:
    alert_manager.send_alerts(alerts)

print("\n✅ Pipeline test complete!")