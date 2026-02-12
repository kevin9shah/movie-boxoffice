import json
import os
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """Get sentiment scores for a text."""
        if not isinstance(text, str):
            return 0.0
        return self.analyzer.polarity_scores(text)['compound']

    def process_youtube_comments(self):
        """Process YouTube comments and calculate average sentiment."""
        input_path = os.path.join(config.RAW_DATA_DIR, "youtube_trailers.json")
        if not os.path.exists(input_path):
            logger.warning(f"File not found: {input_path}")
            return

        with open(input_path, 'r') as f:
            data = json.load(f)

        sentiment_results = []
        
        for item in data:
            movie_id = item.get('movie_id')
            comments = item.get('comments', [])
            
            if not comments:
                avg_sentiment = 0.0
            else:
                scores = [self.analyze_text(c) for c in comments]
                avg_sentiment = sum(scores) / len(scores)
            
            sentiment_results.append({
                "movie_id": movie_id,
                "youtube_sentiment": avg_sentiment,
                "sentiment_volatility": pd.Series(scores).std() if comments else 0.0
            })
            
        # Save results
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "youtube_sentiment.json")
        with open(output_path, 'w') as f:
            json.dump(sentiment_results, f, indent=4)
        logger.info(f"Saved YouTube sentiment to {output_path}")

    def run(self):
        logger.info("Starting Sentiment Analysis...")
        self.process_youtube_comments()
        logger.info("Sentiment Analysis Completed.")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.run()
