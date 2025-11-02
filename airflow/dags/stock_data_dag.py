from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/mayankrajwansh/stock-market-pipeline')

from src.extractors.yahoo_finance import YahooFinanceExtractor
from src.extractors.news_scraper import NewsScraper
from src.transformers.technical_indicators import TechnicalIndicators
from src.transformers.sentiment_analyzer import SentimentAnalyzer
from src.loaders.database import DatabaseManager
from src.alerts.alert_manager import AlertManager
from src.config.settings import settings

# Default arguments for the DAG
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Initialize DAG
dag = DAG(
    'stock_market_data_pipeline',
    default_args=default_args,
    description='Daily stock market data extraction and analysis',
    schedule_interval='0 18 * * 1-5',  # Run at 6 PM on weekdays
    catchup=False,
    tags=['finance', 'stocks', 'etl'],
)

def extract_stock_data(**context):
    """Task 1: Extract stock price data from Yahoo Finance"""
    extractor = YahooFinanceExtractor()
    db = DatabaseManager()
    
    start_time = datetime.now()
    
    try:
        # Get stock data for all Nifty 50 stocks
        symbols = settings.NIFTY_50_STOCKS
        df = extractor.get_multiple_stocks(
            symbols=symbols,
            period=settings.DEFAULT_PERIOD,
            interval=settings.DEFAULT_INTERVAL
        )
        
        if df.empty:
            raise ValueError("No data extracted")
        
        # Push to XCom for next task
        context['task_instance'].xcom_push(key='stock_data_count', value=len(df))
        
        # Store in database
        db.insert_stock_prices(df)
        
        # Log success
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='extract_stock_data',
            status='SUCCESS',
            start_time=start_time,
            end_time=end_time,
            records_processed=len(df)
        )
        
        print(f"Successfully extracted {len(df)} stock price records")
        return len(df)
        
    except Exception as e:
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='extract_stock_data',
            status='FAILED',
            start_time=start_time,
            end_time=end_time,
            error_message=str(e)
        )
        raise

def extract_news_data(**context):
    """Task 2: Extract news and perform sentiment analysis"""
    scraper = NewsScraper()
    analyzer = SentimentAnalyzer()
    db = DatabaseManager()
    
    start_time = datetime.now()
    
    try:
        symbols = settings.NIFTY_50_STOCKS[:10]  # Get news for top 10 stocks
        
        # Scrape news
        articles = scraper.get_news_for_multiple_stocks(symbols)
        
        # Add market news
        market_news = scraper.get_market_news()
        articles.extend(market_news)
        
        if not articles:
            print("No news articles found")
            return 0
        
        # Analyze sentiment
        analyzed_articles = analyzer.analyze_news_articles(articles)
        
        # Store in database
        db.insert_news_sentiment(analyzed_articles)
        
        # Log success
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='extract_news_data',
            status='SUCCESS',
            start_time=start_time,
            end_time=end_time,
            records_processed=len(analyzed_articles)
        )
        
        print(f"Successfully extracted and analyzed {len(analyzed_articles)} articles")
        return len(analyzed_articles)
        
    except Exception as e:
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='extract_news_data',
            status='FAILED',
            start_time=start_time,
            end_time=end_time,
            error_message=str(e)
        )
        print(f"News extraction failed: {str(e)}")
        return 0

def calculate_technical_indicators(**context):
    """Task 3: Calculate technical indicators"""
    calculator = TechnicalIndicators()
    db = DatabaseManager()
    
    start_time = datetime.now()
    
    try:
        symbols = settings.NIFTY_50_STOCKS
        total_processed = 0
        
        for symbol in symbols:
            # Get recent stock data
            df = db.get_latest_stock_data(symbol, days=300)  # Need history for indicators
            
            if df.empty:
                print(f"No data found for {symbol}")
                continue
            
            # Calculate all indicators
            df_with_indicators = calculator.calculate_all_indicators(df)
            
            # Prepare for database
            indicators_df = calculator.prepare_for_database(df_with_indicators)
            
            # Insert into database
            db.insert_technical_indicators(indicators_df)
            
            total_processed += len(indicators_df)
        
        # Log success
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='calculate_technical_indicators',
            status='SUCCESS',
            start_time=start_time,
            end_time=end_time,
            records_processed=total_processed
        )
        
        print(f"Successfully calculated indicators for {total_processed} records")
        return total_processed
        
    except Exception as e:
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='calculate_technical_indicators',
            status='FAILED',
            start_time=start_time,
            end_time=end_time,
            error_message=str(e)
        )
        raise

def check_and_send_alerts(**context):
    """Task 4: Check for alert conditions and send notifications"""
    alert_manager = AlertManager()
    db = DatabaseManager()
    
    start_time = datetime.now()
    
    try:
        symbols = settings.NIFTY_50_STOCKS
        all_alerts = []
        
        for symbol in symbols:
            # Get latest data with indicators
            df = db.get_latest_stock_data(symbol, days=5)
            indicators_df = db.get_latest_indicators(symbol, days=5)
            
            if df.empty or indicators_df.empty:
                continue
            
            # Merge price data with indicators
            merged_df = df.merge(indicators_df, on=['symbol', 'date'], how='inner')
            
            # Check for alerts
            alerts = alert_manager.check_all_alerts(merged_df)
            all_alerts.extend(alerts)
        
        if all_alerts:
            # Store alerts in database
            db.insert_alerts(all_alerts)
            
            # Send alerts
            alert_manager.send_alerts(all_alerts)
        
        # Log success
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='check_and_send_alerts',
            status='SUCCESS',
            start_time=start_time,
            end_time=end_time,
            records_processed=len(all_alerts)
        )
        
        print(f"Checked alerts and found {len(all_alerts)} conditions")
        return len(all_alerts)
        
    except Exception as e:
        end_time = datetime.now()
        db.log_pipeline_execution(
            pipeline_name='check_and_send_alerts',
            status='FAILED',
            start_time=start_time,
            end_time=end_time,
            error_message=str(e)
        )
        raise

# Define tasks
extract_stocks_task = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_stock_data,
    dag=dag,
)

extract_news_task = PythonOperator(
    task_id='extract_news_data',
    python_callable=extract_news_data,
    dag=dag,
)

calculate_indicators_task = PythonOperator(
    task_id='calculate_technical_indicators',
    python_callable=calculate_technical_indicators,
    dag=dag,
)

check_alerts_task = PythonOperator(
    task_id='check_and_send_alerts',
    python_callable=check_and_send_alerts,
    dag=dag,
)

# Define task dependencies
extract_stocks_task >> calculate_indicators_task >> check_alerts_task
extract_news_task >> check_alerts_task