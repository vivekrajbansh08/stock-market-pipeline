import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.transformers.technical_indicators import TechnicalIndicators

@pytest.fixture
def sample_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'symbol': 'TEST',
        'date': dates,
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    return df

def test_rsi_calculation(sample_data):
    """Test RSI indicator calculation"""
    calculator = TechnicalIndicators()
    result = calculator.calculate_rsi(sample_data)
    
    assert 'rsi_14' in result.columns, "RSI column should be created"
    assert result['rsi_14'].notna().any(), "RSI should have values"
    # RSI values should be between 0 and 100
    valid_rsi = result['rsi_14'].dropna()
    assert (valid_rsi >= 0).all(), "RSI should be >= 0"
    assert (valid_rsi <= 100).all(), "RSI should be <= 100"

def test_macd_calculation(sample_data):
    """Test MACD indicator calculation"""
    calculator = TechnicalIndicators()
    result = calculator.calculate_macd(sample_data)
    
    assert 'macd' in result.columns, "MACD column should exist"
    assert 'macd_signal' in result.columns, "MACD signal should exist"
    assert 'macd_histogram' in result.columns, "MACD histogram should exist"

def test_moving_averages(sample_data):
    """Test moving average calculations"""
    calculator = TechnicalIndicators()
    result = calculator.calculate_sma(sample_data, periods=[20, 50])
    
    assert 'sma_20' in result.columns, "SMA 20 should exist"
    assert 'sma_50' in result.columns, "SMA 50 should exist"

def test_all_indicators(sample_data):
    """Test calculation of all indicators"""
    calculator = TechnicalIndicators()
    result = calculator.calculate_all_indicators(sample_data)
    
    expected_columns = ['rsi_14', 'macd', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
    for col in expected_columns:
        assert col in result.columns, f"{col} should be calculated"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])