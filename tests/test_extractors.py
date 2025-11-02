import pytest
import sys
sys.path.append('..')

from src.extractors.yahoo_finance import YahooFinanceExtractor

def test_stock_data_extraction():
    """Test basic stock data extraction"""
    extractor = YahooFinanceExtractor()
    df = extractor.get_stock_data("RELIANCE.NS", period="5d")
    
    assert df is not None, "DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert 'symbol' in df.columns, "Symbol column should exist"
    assert 'close' in df.columns, "Close column should exist"
    assert len(df) > 0, "Should have at least one row"

def test_multiple_stocks():
    """Test extraction for multiple stocks"""
    extractor = YahooFinanceExtractor()
    symbols = ["RELIANCE.NS", "TCS.NS"]
    df = extractor.get_multiple_stocks(symbols, period="5d")
    
    assert not df.empty, "DataFrame should not be empty"
    assert df['symbol'].nunique() == 2, "Should have data for both stocks"

def test_latest_price():
    """Test latest price retrieval"""
    extractor = YahooFinanceExtractor()
    info = extractor.get_latest_price("TCS.NS")
    
    assert info is not None, "Info should not be None"
    assert 'current_price' in info, "Should have current price"
    assert info['current_price'] > 0, "Price should be positive"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])