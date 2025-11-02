import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database operations for stock market data"""
    
    def __init__(self):
        self.connection_string = settings.DATABASE_URL
        self.engine = create_engine(self.connection_string)
        self.logger = logger
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def execute_sql_file(self, sql_file_path: str):
        """Execute SQL file (for initialization)"""
        try:
            with open(sql_file_path, 'r') as file:
                sql = file.read()
            
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Successfully executed {sql_file_path}")
        except Exception as e:
            self.logger.error(f"Error executing SQL file: {str(e)}")
            raise
    
    def insert_stock_prices(self, df: pd.DataFrame):
        """Insert stock price data into database"""
        try:
            # Make a copy and clean the data
            df_to_insert = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
            
            # Convert date to date type (remove timezone)
            df_to_insert['date'] = pd.to_datetime(df_to_insert['date']).dt.date
            
            # Convert numeric columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close']
            for col in numeric_cols:
                df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors='coerce')
            
            # Convert volume to integer
            df_to_insert['volume'] = pd.to_numeric(df_to_insert['volume'], errors='coerce').fillna(0).astype('int64')
            
            # Remove any rows with NaN values
            df_to_insert = df_to_insert.dropna()
            
            if df_to_insert.empty:
                self.logger.warning("No valid rows to insert after cleaning")
                return 0
            
            # Use psycopg2 directly with proper parameterization
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_count = 0
            for _, row in df_to_insert.iterrows():
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
                        str(row['symbol']),
                        row['date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['adj_close']),
                        int(row['volume'])
                    ))
                    insert_count += 1
                except Exception as e:
                    self.logger.warning(f"Error inserting row for {row['symbol']} on {row['date']}: {str(e)}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Successfully inserted {insert_count} stock price records")
            return insert_count
            
        except Exception as e:
            self.logger.error(f"Error inserting stock prices: {str(e)}")
            raise
    
    def insert_technical_indicators(self, df: pd.DataFrame):
        """Insert technical indicators into database"""
        try:
            indicator_columns = [
                'symbol', 'date', 'sma_20', 'sma_50', 'sma_200', 'ema_20',
                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'obv',
                'stochastic_k', 'stochastic_d'
            ]
            
            available_columns = [col for col in indicator_columns if col in df.columns]
            if not available_columns:
                self.logger.info("No indicator columns available to insert")
                return 0
            
            df_to_insert = df[available_columns].copy()
            
            # Convert date to date type (remove timezone)
            df_to_insert['date'] = pd.to_datetime(df_to_insert['date']).dt.date
            
            # Convert all numeric columns to float
            numeric_cols = [col for col in available_columns if col not in ['symbol', 'date']]
            for col in numeric_cols:
                df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors='coerce')
            
            # Remove rows with NaN in critical columns
            df_to_insert = df_to_insert.dropna()
            
            if df_to_insert.empty:
                self.logger.info("No valid technical indicator rows to insert after cleaning")
                return 0
            
            # Remove duplicate (symbol, date) combinations BEFORE inserting
            df_to_insert = df_to_insert.drop_duplicates(subset=['symbol', 'date'], keep='last')
            
            # Use psycopg2 with proper batch insert
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_count = 0
            for _, row in df_to_insert.iterrows():
                try:
                    # Build dynamic query
                    cols = ', '.join(available_columns)
                    placeholders = ', '.join(['%s'] * len(available_columns))
                    update_cols = ', '.join([f"{col} = EXCLUDED.{col}" for col in numeric_cols])
                    
                    query = f"""
                        INSERT INTO technical_indicators ({cols})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, date) DO UPDATE SET {update_cols}
                    """
                    
                    values = tuple(row[col] for col in available_columns)
                    cursor.execute(query, values)
                    insert_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error inserting indicator for {row['symbol']} on {row['date']}: {str(e)}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Successfully inserted {insert_count} technical indicator records")
            return insert_count
            
        except Exception as e:
            self.logger.error(f"Error inserting technical indicators: {str(e)}")
            raise
    
    def insert_trading_signals(self, signals: List[Dict]):
        """Insert trading signals into database"""
        try:
            if not signals:
                return 0
                
            df = pd.DataFrame(signals)
            df.to_sql(
                'trading_signals',
                self.engine,
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Inserted {len(signals)} trading signals")
            return len(signals)
            
        except Exception as e:
            self.logger.error(f"Error inserting trading signals: {str(e)}")
            raise
    
    def insert_alerts(self, alerts: List[Dict]):
        """Insert alerts into database"""
        try:
            if not alerts:
                return 0
                
            df = pd.DataFrame(alerts)
            df.to_sql(
                'alerts',
                self.engine,
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Inserted {len(alerts)} alerts")
            return len(alerts)
            
        except Exception as e:
            self.logger.error(f"Error inserting alerts: {str(e)}")
            raise
    
    def insert_news_sentiment(self, articles: List[Dict]):
        """Insert news sentiment data into database"""
        try:
            if not articles:
                return 0
                
            df = pd.DataFrame(articles)
            columns_to_insert = ['symbol', 'headline', 'source', 'published_date', 
                                'sentiment_score', 'sentiment_label', 'url']
            
            available_columns = [col for col in columns_to_insert if col in df.columns]
            df_to_insert = df[available_columns].copy()
            
            df_to_insert.to_sql(
                'news_sentiment',
                self.engine,
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Inserted {len(df_to_insert)} news sentiment records")
            return len(df_to_insert)
            
        except Exception as e:
            self.logger.error(f"Error inserting news sentiment: {str(e)}")
            raise
    
    def get_latest_stock_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest stock data for a symbol"""
        query = f"""
        SELECT * FROM stock_prices 
        WHERE symbol = '{symbol}' 
        ORDER BY date DESC 
        LIMIT {limit}
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            self.logger.error(f"Error retrieving stock data: {str(e)}")
            return pd.DataFrame()

    def get_latest_indicators(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest technical indicators for a symbol"""
        query = f"""
            SELECT * FROM technical_indicators 
            WHERE symbol = '{symbol}' 
            ORDER BY date DESC 
            LIMIT {limit}
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            self.logger.error(f"Error retrieving indicators: {str(e)}")
            return pd.DataFrame()
    
    def get_pending_alerts(self) -> pd.DataFrame:
        """Retrieve alerts that haven't been sent"""
        query = """
            SELECT * FROM alerts 
            WHERE is_sent = FALSE 
            ORDER BY triggered_at DESC
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            self.logger.error(f"Error retrieving alerts: {str(e)}")
            return pd.DataFrame()
    
    def mark_alert_sent(self, alert_id: int):
        """Mark an alert as sent"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                UPDATE alerts 
                SET is_sent = TRUE, sent_at = %s 
                WHERE id = %s
            """
            cursor.execute(query, (datetime.now(), alert_id))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Marked alert {alert_id} as sent")
            
        except Exception as e:
            self.logger.error(f"Error marking alert as sent: {str(e)}")
            raise
    
    def log_pipeline_execution(self, pipeline_name: str, status: str, 
                               start_time: datetime, end_time: datetime = None,
                               records_processed: int = 0, error_message: str = None):
        """Log pipeline execution details"""
        try:
            duration = None
            if end_time:
                duration = int((end_time - start_time).total_seconds())
            
            data = {
                'pipeline_name': [pipeline_name],
                'status': [status],
                'start_time': [start_time],
                'end_time': [end_time],
                'duration_seconds': [duration],
                'records_processed': [records_processed],
                'error_message': [error_message]
            }
            
            df = pd.DataFrame(data)
            df.to_sql('pipeline_logs', self.engine, if_exists='append', index=False)
            
            self.logger.info(f"Logged pipeline execution: {pipeline_name} - {status}")
            
        except Exception as e:
            self.logger.error(f"Error logging pipeline execution: {str(e)}")

# Example usage
if __name__ == "__main__":
    db = DatabaseManager()
    
    # Test connection
    try:
        conn = db.get_connection()
        print("Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"Database connection failed: {e}")