import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """
    Stock price prediction using Random Forest and LSTM
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = []
        self.logger = logger
    
    def prepare_features(self, df):
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with stock data and technical indicators
        
        Returns:
            Feature matrix and target variable
        """
        df = df.copy()
        
        # Create lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Create rolling features
        df['close_rolling_mean_5'] = df['close'].rolling(window=5).mean()
        df['close_rolling_mean_10'] = df['close'].rolling(window=10).mean()
        df['close_rolling_std_5'] = df['close'].rolling(window=5).std()
        df['volume_rolling_mean_5'] = df['volume'].rolling(window=5).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_1'] = df['price_change'].shift(1)
        df['price_change_2'] = df['price_change'].shift(2)
        
        # Technical indicators (if available)
        technical_features = ['rsi_14', 'macd', 'macd_signal', 'sma_20', 'sma_50']
        
        # Select features
        self.feature_columns = [
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
            'close_rolling_mean_5', 'close_rolling_mean_10', 'close_rolling_std_5',
            'volume_rolling_mean_5',
            'price_change', 'price_change_1', 'price_change_2'
        ]
        
        # Add technical indicators if available
        for col in technical_features:
            if col in df.columns:
                self.feature_columns.append(col)
        
        # Remove rows with NaN
        df = df.dropna()
        
        if df.empty:
            raise ValueError("Not enough data after creating features")
        
        # Features and target
        X = df[self.feature_columns]
        y = df['close']
        
        return X, y
    
    def train_random_forest(self, df, test_size=0.2):
        """
        Train Random Forest model
        
        Args:
            df: DataFrame with stock data
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with model performance metrics
        """
        self.logger.info("Training Random Forest model...")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100,
            'test_mape': np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        }
        
        self.logger.info(f"Model trained - Test RMSE: {metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict_next_days(self, df, days=5):
        """
        Predict next N days
        
        Args:
            df: DataFrame with recent stock data
            days: Number of days to predict
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest first.")
        
        predictions = []
        current_df = df.copy()
        
        for i in range(days):
            # Prepare features for latest data
            X, _ = self.prepare_features(current_df)
            latest_features = X.iloc[-1:].values
            
            # Scale and predict
            latest_scaled = self.scaler.transform(latest_features)
            predicted_price = self.model.predict(latest_scaled)[0]
            
            # Create next day's row
            next_date = current_df['date'].iloc[-1] + pd.Timedelta(days=1)
            next_row = current_df.iloc[-1:].copy()
            next_row['date'] = next_date
            next_row['close'] = predicted_price
            next_row['open'] = predicted_price
            next_row['high'] = predicted_price * 1.01
            next_row['low'] = predicted_price * 0.99
            
            # Append to dataframe
            current_df = pd.concat([current_df, next_row], ignore_index=True)
            
            predictions.append({
                'date': next_date,
                'predicted_price': predicted_price,
                'day': i + 1
            })
        
        return pd.DataFrame(predictions)
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        self.logger.info(f"Model loaded from {filepath}")


# LSTM Model (requires tensorflow)
class LSTMStockPredictor:
    """
    LSTM-based stock price prediction
    Requires: tensorflow
    """
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.logger = logger
    
    def prepare_sequences(self, df):
        """Prepare sequences for LSTM"""
        # Use closing prices
        data = df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_lstm(self, df, epochs=50, batch_size=32):
        """
        Train LSTM model
        Requires tensorflow to be installed
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        self.logger.info("Training LSTM model...")
        
        # Prepare data
        X, y = self.prepare_sequences(df)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_inv, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_inv, test_pred)),
            'train_mae': mean_absolute_error(y_train_inv, train_pred),
            'test_mae': mean_absolute_error(y_test_inv, test_pred)
        }
        
        self.logger.info(f"LSTM trained - Test RMSE: {metrics['test_rmse']:.2f}")
        
        return metrics
    
    def predict_next_days(self, df, days=5):
        """Predict next N days using LSTM"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get last sequence
        data = df['close'].values[-self.sequence_length:].reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        predictions = []
        current_sequence = scaled_data
        
        for i in range(days):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict
            predicted_scaled = self.model.predict(X, verbose=0)
            predicted_price = self.scaler.inverse_transform(predicted_scaled)[0][0]
            
            # Add to predictions
            next_date = df['date'].iloc[-1] + pd.Timedelta(days=i+1)
            predictions.append({
                'date': next_date,
                'predicted_price': predicted_price,
                'day': i + 1
            })
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], predicted_scaled)
            current_sequence = current_sequence.reshape(-1, 1)
        
        return pd.DataFrame(predictions)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'symbol': 'TEST',
        'date': dates,
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Train Random Forest
    predictor = StockPricePredictor()
    metrics = predictor.train_random_forest(df)
    print(f"Model Metrics: {metrics}")
    
    # Predict next 5 days
    predictions = predictor.predict_next_days(df, days=5)
    print(f"\nPredictions:\n{predictions}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\nTop Features:\n{importance.head()}")