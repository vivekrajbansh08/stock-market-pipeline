import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    def __init__(self):
        self.logger = logger
    
    def calculate_sma(self, df: pd.DataFrame, column: str = 'close', periods: list = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Average"""
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        return df
    
    def calculate_ema(self, df: pd.DataFrame, column: str = 'close', periods: list = [20]) -> pd.DataFrame:
        """Calculate Exponential Moving Average"""
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        column: str = 'close', 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        column: str = 'close', 
        period: int = 20, 
        std_dev: int = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['bb_middle'] = df[column].rolling(window=period).mean()
        rolling_std = df[column].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        return df
    
    def calculate_stochastic(
        self, 
        df: pd.DataFrame, 
        period: int = 14, 
        smooth_k: int = 3, 
        smooth_d: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stochastic_k'] = k.rolling(window=smooth_k).mean()
        df['stochastic_d'] = df['stochastic_k'].rolling(window=smooth_d).mean()
        
        return df
    
    def detect_golden_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Golden Cross (SMA50 crosses above SMA200)"""
        df['golden_cross'] = False
        
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            # Detect when SMA50 crosses above SMA200
            prev_below = df['sma_50'].shift(1) < df['sma_200'].shift(1)
            curr_above = df['sma_50'] > df['sma_200']
            df['golden_cross'] = prev_below & curr_above
        
        return df
    
    def detect_death_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Death Cross (SMA50 crosses below SMA200)"""
        df['death_cross'] = False
        
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            # Detect when SMA50 crosses below SMA200
            prev_above = df['sma_50'].shift(1) > df['sma_200'].shift(1)
            curr_below = df['sma_50'] < df['sma_200']
            df['death_cross'] = prev_above & curr_below
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            self.logger.info(f"Calculating indicators for {len(df)} rows")
            
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Sort by date
            df = df.sort_values('date')
            
            # Moving Averages
            df = self.calculate_sma(df, periods=[20, 50, 200])
            df = self.calculate_ema(df, periods=[20])
            
            # Momentum Indicators
            df = self.calculate_rsi(df)
            df = self.calculate_macd(df)
            df = self.calculate_stochastic(df)
            
            # Volatility Indicators
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_atr(df)
            
            # Volume Indicators
            df = self.calculate_obv(df)
            
            # Trading Signals
            df = self.detect_golden_cross(df)
            df = self.detect_death_cross(df)
            
            self.logger.info("All indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    def prepare_for_database(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for database insertion"""
        indicator_columns = [
            'symbol', 'date', 'sma_20', 'sma_50', 'sma_200', 'ema_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'obv',
            'stochastic_k', 'stochastic_d'
        ]
        
        available_columns = [col for col in indicator_columns if col in df.columns]
        return df[available_columns]

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'symbol': 'TEST.NS',
        'date': dates,
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Calculate indicators
    calculator = TechnicalIndicators()
    result = calculator.calculate_all_indicators(sample_data)
    
    print("Calculated Indicators:")
    print(result[['date', 'close', 'sma_20', 'sma_50', 'rsi_14', 'macd']].tail())
    
    # Check for signals
    golden_crosses = result[result['golden_cross'] == True]
    print(f"\nGolden Crosses detected: {len(golden_crosses)}")