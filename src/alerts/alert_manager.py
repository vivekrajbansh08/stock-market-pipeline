import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict
import logging
import pandas as pd

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertManager:
    """Manage and send trading alerts"""
    
    def __init__(self):
        self.logger = logger
        self.rsi_oversold = settings.RSI_OVERSOLD
        self.rsi_overbought = settings.RSI_OVERBOUGHT
        self.price_change_threshold = settings.PRICE_CHANGE_THRESHOLD
    
    def check_rsi_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """Check for RSI-based alerts"""
        alerts = []
        
        if 'rsi_14' not in df.columns:
            return alerts
        
        latest_data = df.iloc[-1]
        symbol = latest_data['symbol']
        rsi = latest_data['rsi_14']
        date = latest_data['date']
        
        if pd.isna(rsi):
            return alerts
        
        # Check for oversold condition
        if rsi < self.rsi_oversold:
            alerts.append({
                'symbol': symbol,
                'alert_type': 'RSI_OVERSOLD',
                'condition_met': f'RSI below {self.rsi_oversold}',
                'value': rsi,
                'triggered_at': datetime.now()
            })
            self.logger.info(f"RSI Oversold alert for {symbol}: {rsi:.2f}")
        
        # Check for overbought condition
        elif rsi > self.rsi_overbought:
            alerts.append({
                'symbol': symbol,
                'alert_type': 'RSI_OVERBOUGHT',
                'condition_met': f'RSI above {self.rsi_overbought}',
                'value': rsi,
                'triggered_at': datetime.now()
            })
            self.logger.info(f"RSI Overbought alert for {symbol}: {rsi:.2f}")
        
        return alerts
    
    def check_golden_cross(self, df: pd.DataFrame) -> List[Dict]:
        """Check for Golden Cross pattern"""
        alerts = []
        
        if 'golden_cross' not in df.columns:
            return alerts
        
        golden_cross_signals = df[df['golden_cross'] == True]
        
        for _, row in golden_cross_signals.iterrows():
            alerts.append({
                'symbol': row['symbol'],
                'alert_type': 'GOLDEN_CROSS',
                'condition_met': 'SMA50 crossed above SMA200',
                'value': row.get('sma_50', 0),
                'triggered_at': datetime.now()
            })
            self.logger.info(f"Golden Cross alert for {row['symbol']}")
        
        return alerts
    
    def check_death_cross(self, df: pd.DataFrame) -> List[Dict]:
        """Check for Death Cross pattern"""
        alerts = []
        
        if 'death_cross' not in df.columns:
            return alerts
        
        death_cross_signals = df[df['death_cross'] == True]
        
        for _, row in death_cross_signals.iterrows():
            alerts.append({
                'symbol': row['symbol'],
                'alert_type': 'DEATH_CROSS',
                'condition_met': 'SMA50 crossed below SMA200',
                'value': row.get('sma_50', 0),
                'triggered_at': datetime.now()
            })
            self.logger.info(f"Death Cross alert for {row['symbol']}")
        
        return alerts
    
    def check_macd_crossover(self, df: pd.DataFrame) -> List[Dict]:
        """Check for MACD crossover"""
        alerts = []
        
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return alerts
        
        if len(df) < 2:
            return alerts
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Bullish crossover (MACD crosses above signal)
        if (previous['macd'] < previous['macd_signal'] and 
            latest['macd'] > latest['macd_signal']):
            alerts.append({
                'symbol': latest['symbol'],
                'alert_type': 'MACD_BULLISH_CROSSOVER',
                'condition_met': 'MACD crossed above signal line',
                'value': latest['macd'],
                'triggered_at': datetime.now()
            })
        
        # Bearish crossover (MACD crosses below signal)
        elif (previous['macd'] > previous['macd_signal'] and 
              latest['macd'] < latest['macd_signal']):
            alerts.append({
                'symbol': latest['symbol'],
                'alert_type': 'MACD_BEARISH_CROSSOVER',
                'condition_met': 'MACD crossed below signal line',
                'value': latest['macd'],
                'triggered_at': datetime.now()
            })
        
        return alerts
    
    def check_price_change(self, df: pd.DataFrame) -> List[Dict]:
        """Check for significant price changes"""
        alerts = []
        
        if len(df) < 2:
            return alerts
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        price_change_pct = ((latest['close'] - previous['close']) / previous['close']) * 100
        
        if abs(price_change_pct) >= self.price_change_threshold:
            alert_type = 'PRICE_SURGE' if price_change_pct > 0 else 'PRICE_DROP'
            alerts.append({
                'symbol': latest['symbol'],
                'alert_type': alert_type,
                'condition_met': f'Price change: {price_change_pct:.2f}%',
                'value': price_change_pct,
                'triggered_at': datetime.now()
            })
        
        return alerts
    
    def check_all_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """Check all alert conditions"""
        all_alerts = []
        
        all_alerts.extend(self.check_rsi_alerts(df))
        all_alerts.extend(self.check_golden_cross(df))
        all_alerts.extend(self.check_death_cross(df))
        all_alerts.extend(self.check_macd_crossover(df))
        all_alerts.extend(self.check_price_change(df))
        
        return all_alerts
    
    def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        try:
            if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
                self.logger.warning("Email credentials not configured")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USERNAME
            msg['To'] = settings.ALERT_EMAIL
            msg['Subject'] = f"Trading Alert: {alert['symbol']} - {alert['alert_type']}"
            
            body = f"""
            Stock: {alert['symbol']}
            Alert Type: {alert['alert_type']}
            Condition: {alert['condition_met']}
            Value: {alert['value']:.2f}
            Time: {alert['triggered_at']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            
            text = msg.as_string()
            server.sendmail(settings.SMTP_USERNAME, settings.ALERT_EMAIL, text)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert['symbol']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def print_alert(self, alert: Dict):
        """Print alert to console"""
        print(f"\n{'='*60}")
        print(f"🚨 TRADING ALERT 🚨")
        print(f"{'='*60}")
        print(f"Symbol: {alert['symbol']}")
        print(f"Alert Type: {alert['alert_type']}")
        print(f"Condition: {alert['condition_met']}")
        print(f"Value: {alert['value']:.2f}")
        print(f"Time: {alert['triggered_at']}")
        print(f"{'='*60}\n")
    
    def send_alerts(self, alerts: List[Dict]):
        """Send all alerts"""
        for alert in alerts:
            # Always print to console
            self.print_alert(alert)
            
            # Try to send email if configured
            if settings.ALERT_EMAIL:
                self.send_email_alert(alert)

# Example usage
if __name__ == "__main__":
    alert_manager = AlertManager()
    
    # Create sample alert
    sample_alert = {
        'symbol': 'RELIANCE.NS',
        'alert_type': 'RSI_OVERSOLD',
        'condition_met': 'RSI below 30',
        'value': 28.5,
        'triggered_at': datetime.now()
    }
    
    alert_manager.print_alert(sample_alert)