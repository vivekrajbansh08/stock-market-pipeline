import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraper:
    """Scrape financial news from various sources"""
    
    def __init__(self):
        self.logger = logger
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_google_finance_news(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Google Finance RSS feed
        
        Args:
            symbol: Stock ticker symbol (without exchange suffix)
        
        Returns:
            List of news articles
        """
        try:
            # Remove exchange suffix for Google Finance
            clean_symbol = symbol.split('.')[0]
            
            url = f"https://news.google.com/rss/search?q={clean_symbol}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:10]:  # Get top 10 articles
                article = {
                    'symbol': symbol,
                    'headline': entry.title,
                    'source': entry.source.title if hasattr(entry, 'source') else 'Google News',
                    'published_date': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                    'url': entry.link,
                }
                articles.append(article)
            
            self.logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def get_moneycontrol_news(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Moneycontrol
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            List of news articles
        """
        try:
            # This is a simplified version - Moneycontrol scraping may require more robust handling
            clean_symbol = symbol.split('.')[0].lower()
            url = f"https://www.moneycontrol.com/news/tags/{clean_symbol}.html"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            news_items = soup.find_all('li', class_='clearfix', limit=10)
            
            for item in news_items:
                title_tag = item.find('h2')
                link_tag = item.find('a')
                
                if title_tag and link_tag:
                    article = {
                        'symbol': symbol,
                        'headline': title_tag.get_text(strip=True),
                        'source': 'Moneycontrol',
                        'published_date': datetime.now(),  # Would need to parse actual date
                        'url': link_tag['href'] if link_tag.has_attr('href') else '',
                    }
                    articles.append(article)
            
            self.logger.info(f"Fetched {len(articles)} articles from Moneycontrol for {symbol}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Moneycontrol news for {symbol}: {str(e)}")
            return []
    
    def get_news_for_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """
        Fetch news for multiple stock symbols
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Combined list of news articles
        """
        all_articles = []
        
        for symbol in symbols:
            articles = self.get_google_finance_news(symbol)
            all_articles.extend(articles)
        
        self.logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles
    
    def get_market_news(self) -> List[Dict]:
        """
        Fetch general market news from Economic Times RSS
        
        Returns:
            List of general market news articles
        """
        try:
            url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:15]:
                article = {
                    'symbol': 'MARKET',
                    'headline': entry.title,
                    'source': 'Economic Times',
                    'published_date': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                    'url': entry.link,
                }
                articles.append(article)
            
            self.logger.info(f"Fetched {len(articles)} general market articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching market news: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    scraper = NewsScraper()
    
    # Test single stock news
    articles = scraper.get_google_finance_news("RELIANCE.NS")
    if articles:
        print(f"Sample article:")
        print(articles[0])
    
    # Test multiple stocks
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    all_articles = scraper.get_news_for_multiple_stocks(symbols)
    print(f"\nTotal articles: {len(all_articles)}")
    
    # Test market news
    market_news = scraper.get_market_news()
    print(f"\nMarket news count: {len(market_news)}")