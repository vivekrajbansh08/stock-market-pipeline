import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceExtractor:
    """Extract stock market data from Yahoo Finance"""
    
    def __init__(self):
        self.logger = logger
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., 'RELIANCE.NS')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching data for {symbol} with period={period}, interval={interval}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Rename columns to match our database schema
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Keep only relevant columns
            df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Add adj_close (same as close for now)
            df['adj_close'] = df['close']
            
            self.logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stock symbols
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Combined DataFrame with all stocks
        """
        all_data = []
        
        for symbol in symbols:
            df = self.get_stock_data(symbol, period, interval)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            self.logger.warning("No data fetched for any symbols")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def get_latest_price(self, symbol: str) -> Optional[dict]:
        """
        Get the latest price and related info for a symbol
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with latest price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
            }
        except Exception as e:
            self.logger.error(f"Error fetching latest price for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol: str) -> Optional[dict]:
        """
        Get company information for a symbol
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with company info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                'employees': info.get('fullTimeEmployees'),
            }
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    extractor = YahooFinanceExtractor()
    
    # Test single stock
    df = extractor.get_stock_data("RELIANCE.NS", period="1mo")
    if df is not None:
        print(df.head())
    
    # Test multiple stocks
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    combined_df = extractor.get_multiple_stocks(symbols, period="5d")
    print(f"\nCombined data:\n{combined_df.head()}")
    
    # Test latest price
    latest = extractor.get_latest_price("RELIANCE.NS")
    print(f"\nLatest price info:\n{latest}")