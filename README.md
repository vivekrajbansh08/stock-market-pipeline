# 📈 Stock Market Data Pipeline

A comprehensive, production-ready financial data pipeline for automated stock analytics, technical indicators, sentiment analysis, and AI-powered price predictions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)
![Airflow](https://img.shields.io/badge/Airflow-2.7+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- ✅ **Automated Data Extraction** - Yahoo Finance API integration
- ✅ **12+ Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages, etc.
- ✅ **Sentiment Analysis** - News scraping with VADER sentiment scoring
- ✅ **Trading Alerts** - 7 types of real-time notifications
- ✅ **AI Price Predictions** - Random Forest & LSTM models
- ✅ **Interactive Dashboard** - 5 Streamlit pages with advanced visualizations
- ✅ **Portfolio Tracking** - Track investments with profit/loss calculations
- ✅ **Strategy Backtesting** - Test trading strategies on historical data
- ✅ **Workflow Orchestration** - Apache Airflow for automated daily runs

## 📊 Dashboard Preview

The application includes 5 interactive pages:
1. **Main Dashboard** - Real-time price charts and key metrics
2. **Technical Analysis** - Advanced charting with indicators
3. **Portfolio Tracker** - Investment tracking and performance
4. **Backtesting** - Strategy testing with historical data
5. **Alerts** - Trading signal monitoring
6. **AI Predictions** - ML-powered price forecasts

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Extraction** | Python, yfinance, BeautifulSoup, feedparser |
| **Transformation** | pandas, NumPy, ta, pandas-ta, TextBlob, VADER |
| **Loading** | PostgreSQL, SQLAlchemy, psycopg2 |
| **Orchestration** | Apache Airflow |
| **Visualization** | Streamlit, Plotly, Matplotlib |
| **ML Models** | scikit-learn, Random Forest |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-market-pipeline.git
cd stock-market-pipeline


