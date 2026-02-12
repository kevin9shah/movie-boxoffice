import os
import sys
import logging
import json
import time
from googlesearch import search

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GoogleSearchCollector:
    def __init__(self):
        pass

    def get_search_snippets(self, query, num_results=5):
        """Fetch search result snippets for a query."""
        try:
            # search() yields URLs. We need a way to get snippets.
            # standard googlesearch-python only returns URLs.
            # To get snippets without a paid API is hard reliably.
            # However, for this task, the user asked to "scrap google".
            # We will use the 'advanced' mode if available or just collect URLs
            # and potentially scrape them. 
            # Given the constraints and to avoid IP bans, we might just assume 
            # we can get the title/desc if the library supports it, or valid URLs.
            # Update: googlesearch-python is simple. 
            # Let's try to get the URLs and maybe we can use a library 
            # like 'trafilatura' or just 'requests' to get text from the pages?
            # NO, that's too heavy.
            # The USER wants to fix the 0.0 sentiment using Google data.
            # A simple approach: Use the 'search_web' tool equivalent logic? 
            # But I am writing python code for the user to run.
            # I will use 'googlesearch' to get URLs and maybe just return dummy snippets 
            # or try to fetch the page title as "text" for now if we can't scrape SERP.
            
            # actually, let's use a simulated approach for the "snippet" if the lib doesn't giving it,
            # OR better: The user wants to FIX the 0.0 sentiment.
            # I will construct a search query "Movie Title reviews"
            # and collect the URLs. 
            
            results = []
            for url in search(query, num_results=num_results, advanced=True):
                # advanced=True yields SearchResult objects with title, description, url
                results.append({
                    "title": url.title,
                    "description": url.description,
                    "url": url.url
                })
                time.sleep(2) # be gentle
            return results
        except Exception as e:
            logger.error(f"Error searching Google for {query}: {e}")
            return []

    def save_data(self, data, filename):
        """Save data to JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def run(self, movie_list):
        logger.info("Starting Google Search Data Collection...")
        all_data = []

        for movie in movie_list:
            title = movie.get('title')
            movie_id = movie.get('id')
            
            # Check if this movie needs Google data (e.g. if we know it has 0 sentiment)
            # For now, let's just do it for all or a subset?
            # To avoid massive time, maybe strict it? 
            # The user said "scrap google ALSO", implying for all.
            # But query limits... let's try for the list.
            
            logger.info(f"Searching Google for: {title}")
            query = f"{title} movie reviews sentiment"
            snippets = self.get_search_snippets(query, num_results=3)
            
            if snippets:
                all_data.append({
                    "movie_id": movie_id,
                    "title": title,
                    "google_snippets": snippets
                })
            
        self.save_data(all_data, "google_sentiment.json")
        logger.info("Google Search Collection Completed.")

if __name__ == "__main__":
    # Test with Zootopia 2
    collector = GoogleSearchCollector()
    collector.run([{"id": 1084242, "title": "Zootopia 2"}])
