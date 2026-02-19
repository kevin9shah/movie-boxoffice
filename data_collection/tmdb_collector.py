import requests
import os
import json
import logging
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TMDBCollector:
    def __init__(self):
        self.api_key = config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json;charset=utf-8"
        }

    def fetch_popular_movies(self, page=1):
        """Fetch a list of popular movies."""
        url = f"{self.base_url}/movie/popular"
        params = {
            "api_key": self.api_key,
            "language": "en-US",
            "page": page
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching popular movies: {e}")
            return []

    def fetch_movie_details(self, movie_id):
        """Fetch detailed metadata for a specific movie."""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            "api_key": self.api_key,
            "language": "en-US",
            "append_to_response": "credits,keywords,release_dates"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching details for movie ID {movie_id}: {e}")
            return None

    def extract_cast_and_crew(self, movie_details):
        """Extract cast and crew information from movie details."""
        cast_crew = {
            "cast": [],
            "directors": [],
            "composers": []
        }
        
        if not movie_details:
            return cast_crew
        
        # Extract cast (top 5 actors)
        if 'credits' in movie_details and 'cast' in movie_details['credits']:
            cast_crew['cast'] = [
                {
                    "name": actor['name'],
                    "character": actor.get('character', ''),
                    "popularity": actor.get('popularity', 0)
                }
                for actor in movie_details['credits']['cast'][:5]
            ]
        
        # Extract directors
        if 'credits' in movie_details and 'crew' in movie_details['credits']:
            directors = [
                {
                    "name": crew['name'],
                    "popularity": crew.get('popularity', 0)
                }
                for crew in movie_details['credits']['crew']
                if crew.get('job') == 'Director'
            ]
            cast_crew['directors'] = directors
            
            # Extract music composers
            composers = [
                {
                    "name": crew['name'],
                    "popularity": crew.get('popularity', 0)
                }
                for crew in movie_details['credits']['crew']
                if crew.get('job') == 'Original Music Composer'
            ]
            cast_crew['composers'] = composers
        
        return cast_crew

    def save_data(self, data, filename):
        """Save data to JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def run(self, pages=1):
        """Main execution method."""
        logger.info("Starting TMDB Data Collection...")
        all_movies = []
        cast_crew_data = {}
        
        for page in range(1, pages + 1):
            logger.info(f"Fetching page {page}...")
            movies = self.fetch_popular_movies(page)
            
            for movie in movies:
                movie_id = movie['id']
                details = self.fetch_movie_details(movie_id)
                if details:
                    all_movies.append(details)
                    
                    # Extract and save cast/crew data
                    cast_crew = self.extract_cast_and_crew(details)
                    cast_crew_data[str(movie_id)] = {
                        "title": details.get('title', ''),
                        "cast": cast_crew['cast'],
                        "directors": cast_crew['directors'],
                        "composers": cast_crew['composers']
                    }
        
        self.save_data(all_movies, "tmdb_movies.json")
        self.save_data(cast_crew_data, "tmdb_cast_crew.json")
        logger.info(f"Completed! Collected {len(all_movies)} movies.")

if __name__ == "__main__":
    collector = TMDBCollector()
    collector.run(pages=1) # Fetch 1 page for testing
