from pytrends.request import TrendReq
import os
import json
import logging
import sys
import time
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrendsCollector:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)

    def fetch_interest_over_time(self, keywords):
        """Fetch interest over time for a list of keywords (max 5 at a time)."""
        try:
            self.pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo='', gprop='')
            data = self.pytrends.interest_over_time()
            if not data.empty:
                data = data.drop(columns=['isPartial'], errors='ignore')
                data.index = data.index.astype(str)
                return data.to_dict() # Convert DataFrame to dict for JSON serialization
            return {}
        except Exception as e:
            logger.error(f"Error fetching trends for {keywords}: {e}")
            return {}

    def save_data(self, data, filename):
        """Save data to JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            # Convert timestamp keys to strings if necessary
            # Simple dump might fail on Timestamp objects if not handled, 
            # but to_dict() usually handles it or gives timestamps as keys.
            # We'll need a custom encoder or convert to string if keys are timestamps.
            # Here we assume standard dict structure.
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, default=str)
            logger.info(f"Data saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def run(self, movie_list):
        """Main execution method."""
        logger.info("Starting Google Trends Data Collection...")
        all_trends_data = {}
        
        # Google Trends allows max 5 keywords per request. 
        # Ideally we query one by one or in batches of 5.
        # For simplicity and to avoid rate limits, we'll do one by one with sleep.
        
        for movie in movie_list:
            title = movie.get('title')
            logger.info(f"Fetching trends for: {title}")
            trends = self.fetch_interest_over_time([title])
            
            if trends:
                 # The structure from to_dict() is {keyword: {timestamp: value, ...}}
                 # We want to associate it with the movie.
                 all_trends_data[title] = trends.get(title, {})
            
            # Sleep to avoid rate limiting
            time.sleep(2) 
            
        self.save_data(all_trends_data, "google_trends.json")
        logger.info(f"Completed! Collected trends for {len(all_trends_data)} movies.")

if __name__ == "__main__":
    # Dummy data
    dummy_movies = [{"title": "Avatar"}, {"title": "Titanic"}]
    collector = TrendsCollector()
    collector.run(dummy_movies)
