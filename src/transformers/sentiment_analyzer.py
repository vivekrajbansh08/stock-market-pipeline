from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment of financial news"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.logger = logger
    
    def analyze_with_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': polarity,
                'label': label,
                'method': 'textblob'
            }
        except Exception as e:
            self.logger.error(f"TextBlob error: {str(e)}")
            return {'score': 0, 'label': 'neutral', 'method': 'textblob'}
    
    def analyze_with_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']  # -1 to 1
            
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': compound,
                'label': label,
                'method': 'vader'
            }
        except Exception as e:
            self.logger.error(f"VADER error: {str(e)}")
            return {'score': 0, 'label': 'neutral', 'method': 'vader'}
    
    def analyze_news_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of news articles
        
        Args:
            articles: List of article dictionaries with 'headline' key
        
        Returns:
            Articles with sentiment scores added
        """
        analyzed_articles = []
        
        for article in articles:
            headline = article.get('headline', '')
            
            if not headline:
                continue
            
            # Use VADER for financial sentiment (it's better for social media/news)
            sentiment = self.analyze_with_vader(headline)
            
            article['sentiment_score'] = sentiment['score']
            article['sentiment_label'] = sentiment['label']
            
            analyzed_articles.append(article)
        
        self.logger.info(f"Analyzed sentiment for {len(analyzed_articles)} articles")
        return analyzed_articles
    
    def aggregate_sentiment(self, articles: List[Dict], symbol: str = None) -> Dict:
        """
        Aggregate sentiment scores for a stock or overall market
        
        Args:
            articles: List of analyzed articles
            symbol: Optional stock symbol to filter by
        
        Returns:
            Aggregated sentiment metrics
        """
        if symbol:
            filtered_articles = [a for a in articles if a.get('symbol') == symbol]
        else:
            filtered_articles = articles
        
        if not filtered_articles:
            return {
                'symbol': symbol,
                'avg_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }
        
        scores = [a.get('sentiment_score', 0) for a in filtered_articles]
        labels = [a.get('sentiment_label', 'neutral') for a in filtered_articles]
        
        return {
            'symbol': symbol,
            'avg_sentiment': sum(scores) / len(scores) if scores else 0,
            'positive_count': labels.count('positive'),
            'negative_count': labels.count('negative'),
            'neutral_count': labels.count('neutral'),
            'total_articles': len(filtered_articles)
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    sample_articles = [
        {'symbol': 'RELIANCE.NS', 'headline': 'Reliance Industries reports record profits!'},
        {'symbol': 'RELIANCE.NS', 'headline': 'Concerns over Reliance debt levels'},
        {'symbol': 'TCS.NS', 'headline': 'TCS wins major international contract'},
    ]
    
    analyzed = analyzer.analyze_news_articles(sample_articles)
    for article in analyzed:
        print(f"{article['headline']}: {article['sentiment_label']} ({article['sentiment_score']:.3f})")
    
    agg = analyzer.aggregate_sentiment(analyzed, 'RELIANCE.NS')
    print(f"\nRELIANCE.NS Sentiment: {agg}")