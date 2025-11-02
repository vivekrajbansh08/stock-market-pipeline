"""
Initial Data Load Script
This script loads historical data for the first time before starting the regular pipeline
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.yahoo_finance import YahooFinanceExtractor
from src.extractors.news_scraper import NewsScraper
from src.transformers.technical_indicators import TechnicalIndicators
from src.transformers.sentiment_analyzer import SentimentAnalyzer
from src.loaders.database import DatabaseManager
from src.config.settings import settings
from datetime import datetime
import time

def print_banner():
    """Print a nice banner"""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║     📈 Stock Market Data Pipeline                 ║
    ║        Initial Data Load Script                  ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)

def initial_load():
    """Run initial data load"""
    print_banner()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize components
    print("🔧 Initializing components...")
    extractor = YahooFinanceExtractor()
    scraper = NewsScraper()
    calculator = TechnicalIndicators()
    analyzer = SentimentAnalyzer()
    db = DatabaseManager()
    print("✓ Components initialized\n")
    
# Get stocks to process
# Get stocks to process
symbols = settings.NIFTY_50_STOCKS

# Ensure ^NSEI (Nifty 50 Index) is always included
if "^NSEI" not in symbols:
    symbols = ["^NSEI"] + symbols

print(f"📊 Processing {len(symbols)} symbols (including NIFTY 50 index)...\n")

# =================================================================
# STEP 1: Extract and Load Stock Price Data
# =================================================================
print("=" * 60)
print("STEP 1: Extracting Historical Stock Prices")
print("=" * 60)

all_price_data = []
successful_extractions = 0

for i, symbol in enumerate(symbols, 1):
    try:
        print(f"[{i}/{len(symbols)}] Fetching {symbol}...", end=" ")
        
        # Fetch 6 months of daily data
        df = extractor.get_stock_data(symbol, period="6mo", interval="1d")
        
        if df is not None and not df.empty:
            all_price_data.append(df)
            successful_extractions += 1
            print(f"✓ ({len(df)} records)")
        else:
            print("✗ (No data)")
        
        # Rate limiting - be nice to Yahoo Finance
        time.sleep(0.5)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        continue

print(f"\n✓ Successfully extracted data for {successful_extractions}/{len(symbols)} stocks")

# Combine and save all price data
if all_price_data:
    import pandas as pd
    combined_df = pd.concat(all_price_data, ignore_index=True)
    print(f"💾 Saving {len(combined_df)} price records to database...")
    db.insert_stock_prices(combined_df)
    print("✓ Price data saved\n")
    
    # =================================================================
    # STEP 2: Calculate and Load Technical Indicators
    # =================================================================
    print("=" * 60)
    print("STEP 2: Calculating Technical Indicators")
    print("=" * 60)
    
    total_indicators = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] Processing {symbol}...", end=" ")
            
            # Get the price data we just loaded
            stock_df = combined_df[combined_df['symbol'] == symbol].copy()
            
            if stock_df.empty:
                print("✗ (No price data)")
                continue
            
            # Calculate all indicators
            indicators_df = calculator.calculate_all_indicators(stock_df)
            
            # Prepare for database
            prepared_df = calculator.prepare_for_database(indicators_df)
            
            # Save to database
            db.insert_technical_indicators(prepared_df)
            
            total_indicators += len(prepared_df)
            print(f"✓ ({len(prepared_df)} records)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    print(f"\n✓ Calculated and saved {total_indicators} indicator records\n")
    
    # =================================================================
    # STEP 3: Extract and Analyze News
    # =================================================================
    print("=" * 60)
    print("STEP 3: Extracting and Analyzing News")
    print("=" * 60)
    
    # Get news for top 10 stocks to avoid rate limiting
    news_symbols = symbols[:10]
    print(f"Fetching news for: {', '.join(news_symbols)}\n")
    
    all_articles = []
    
    # Get stock-specific news
    for symbol in news_symbols:
        try:
            print(f"Fetching news for {symbol}...", end=" ")
            articles = scraper.get_google_finance_news(symbol)
            all_articles.extend(articles)
            print(f"✓ ({len(articles)} articles)")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    # Get general market news
    print("\nFetching general market news...", end=" ")
    try:
        market_news = scraper.get_market_news()
        all_articles.extend(market_news)
        print(f"✓ ({len(market_news)} articles)")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    # Analyze sentiment
    if all_articles:
        print(f"\n📰 Analyzing sentiment for {len(all_articles)} articles...")
        analyzed_articles = analyzer.analyze_news_articles(all_articles)
        
        # Save to database
        db.insert_news_sentiment(analyzed_articles)
        
        # Show sentiment summary
        positive = sum(1 for a in analyzed_articles if a.get('sentiment_label') == 'positive')
        negative = sum(1 for a in analyzed_articles if a.get('sentiment_label') == 'negative')
        neutral = sum(1 for a in analyzed_articles if a.get('sentiment_label') == 'neutral')
        
        print(f"✓ Sentiment Analysis: {positive} positive, {neutral} neutral, {negative} negative\n")
    
    # =================================================================
    # STEP 4: Generate Initial Alerts
    # =================================================================
    print("=" * 60)
    print("STEP 4: Checking for Trading Alerts")
    print("=" * 60)
    
    from src.alerts.alert_manager import AlertManager
    alert_manager = AlertManager()
    
    all_alerts = []
    
    for symbol in symbols:
        try:
            # Get recent data
            price_df = db.get_latest_stock_data(symbol, days=5)
            indicators_df = db.get_latest_indicators(symbol, days=5)
            
            if price_df.empty or indicators_df.empty:
                continue
            
            # Merge dataframes
            merged_df = price_df.merge(indicators_df, on=['symbol', 'date'], how='inner')
            
            # Check for alerts
            alerts = alert_manager.check_all_alerts(merged_df)
            
            if alerts:
                print(f"⚠️  {symbol}: {len(alerts)} alert(s) found")
                all_alerts.extend(alerts)
            
        except Exception as e:
            continue
    
    # Save and display alerts
    if all_alerts:
        db.insert_alerts(all_alerts)
        print(f"\n🚨 Total alerts found: {len(all_alerts)}")
        print("\nAlert Summary:")
        
        from collections import Counter
        alert_types = Counter(a['alert_type'] for a in all_alerts)
        for alert_type, count in alert_types.most_common():
            print(f"  - {alert_type}: {count}")
        
        print("\n💡 Tip: Check the dashboard to see detailed alert information")
    else:
        print("\n✓ No alerts triggered at this time")
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print(f"✓ Stocks Processed: {successful_extractions}/{len(symbols)}")
    print(f"✓ Price Records: {len(combined_df) if all_price_data else 0}")
    print(f"✓ Indicator Records: {total_indicators}")
    print(f"✓ News Articles: {len(all_articles)}")
    print(f"✓ Alerts Generated: {len(all_alerts)}")
    print("=" * 60)
    
    print(f"\n✅ Initial data load completed successfully!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Next Steps:")
    print("  1. Start the Streamlit dashboard: streamlit run dashboard/streamlit_app.py")
    print("  2. Start Airflow for scheduled updates (optional)")
    print("  3. Check the dashboard at http://localhost:8501")
    print("\n🎉 Happy Trading!\n")

if __name__ == "__main__":
    try:
        initial_load()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during initial load: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)