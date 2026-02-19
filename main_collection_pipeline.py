import os
import sys
import logging
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from data_collection.tmdb_collector import TMDBCollector
from data_collection.youtube_collector import YouTubeCollector
from data_collection.trends_collector import TrendsCollector
from data_collection.wikipedia_collector import WikipediaBoxOfficeCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Main Data Collection Pipeline...")

    # 1. Fetch Movies from TMDB (includes cast/crew extraction)
    logger.info("Collecting TMDB data (movies, cast, crew)...")
    tmdb = TMDBCollector()
    # In a real run, you might want more pages or a specific logic.
    # Fetch 5 pages (~100 movies) as requested.
    tmdb.run(pages=5) 
    
    # Load the collected movies to pass to other collectors
    tmdb_file = os.path.join(config.RAW_DATA_DIR, "tmdb_movies.json")
    if not os.path.exists(tmdb_file):
        logger.error("TMDB data not found. Aborting.")
        return

    with open(tmdb_file, 'r') as f:
        movies = json.load(f)
        
    # Simplify movie list for other collectors (id and title)
    movie_list = [{"id": m.get("id"), "title": m.get("title")} for m in movies]
    
    # 2. Fetch Wikipedia Box Office Data (NEW)
    logger.info("Collecting Wikipedia box office data...")
    wikipedia = WikipediaBoxOfficeCollector()
    wikipedia.run(movie_data_file="tmdb_movies.json")
    
    # 3. Fetch YouTube Data
    logger.info("Collecting YouTube trailer data...")
    youtube = YouTubeCollector()
    youtube.run(movie_list)

    # 4. Fetch Google Trends Data
    logger.info("Collecting Google Trends data...")
    trends = TrendsCollector()
    trends.run(movie_list)

    # 5. Fetch Google Search Data (for sentiment augmentation)
    # google_search = GoogleSearchCollector()
    # google_search.run(movie_list) # Commented out by default to save time/quota, enable if needed.
    
    logger.info("Main Data Collection Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
