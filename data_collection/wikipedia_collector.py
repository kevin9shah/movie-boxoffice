import requests
import os
import json
import logging
import sys
import re
from bs4 import BeautifulSoup

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WikipediaBoxOfficeCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search_movie(self, title, year=None):
        """Search for a movie on Wikipedia."""
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "format": "json",
                "srsearch": f"{title} film" if year is None else f"{title} {year} film",
                "srlimit": 5
            }
            
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                search_results = data['query']['search']
                if search_results:
                    return search_results[0]['title']  # Return first match
            
            return None
        except Exception as e:
            logger.error(f"Error searching Wikipedia for {title}: {e}")
            return None

    def get_page_content(self, page_title):
        """Get the content of a Wikipedia page."""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "titles": page_title,
                "prop": "extracts",
                "explaintext": True,
                "redirects": 1
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            first_page = next(iter(pages.values()))
            
            if 'extract' in first_page:
                return first_page['extract']
            
            return None
        except Exception as e:
            logger.error(f"Error fetching page content for {page_title}: {e}")
            return None

    def extract_box_office_info(self, content):
        """Extract box office information from Wikipedia content."""
        box_office_data = {
            "worldwide_box_office": None,
            "domestic_box_office": None,
            "budget": None
        }
        
        if not content:
            return box_office_data
        
        # Parse lines for box office info
        lines = content.split('\n')
        for line in lines:
            # Look for box office patterns
            if 'box office' in line.lower():
                # Try to find numbers with $ and million/billion
                numbers = re.findall(r'\$[\d,]+(?:\s*million|\s*billion)?', line, re.IGNORECASE)
                if numbers:
                    box_office_data['worldwide_box_office'] = numbers[-1] if numbers else None
            
            if 'budget' in line.lower() and '$' in line:
                numbers = re.findall(r'\$[\d,]+(?:\s*million|\s*billion)?', line, re.IGNORECASE)
                if numbers:
                    box_office_data['budget'] = numbers[0]
        
        return box_office_data

    def parse_monetary_value(self, value_str):
        """Convert monetary string to float (in millions)."""
        if not value_str:
            return None
        
        try:
            # Remove $ and commas
            cleaned = value_str.replace('$', '').replace(',', '').strip()
            
            # Handle million/billion
            if 'billion' in cleaned.lower():
                cleaned = cleaned.lower().replace('billion', '').strip()
                return float(cleaned) * 1000
            elif 'million' in cleaned.lower():
                cleaned = cleaned.lower().replace('million', '').strip()
                return float(cleaned)
            else:
                return float(cleaned) / 1_000_000  # Convert to millions
        except:
            return None

    def fetch_movie_box_office(self, title, year=None):
        """Fetch box office data for a movie from Wikipedia."""
        logger.info(f"Fetching Wikipedia box office data for: {title}")
        
        # Search for the movie
        page_title = self.search_movie(title, year)
        if not page_title:
            logger.warning(f"Could not find Wikipedia page for {title}")
            return None
        
        # Get content
        content = self.get_page_content(page_title)
        if not content:
            logger.warning(f"Could not fetch content for {page_title}")
            return None
        
        # Extract box office info
        box_office_data = self.extract_box_office_info(content)
        
        # Parse monetary values
        result = {
            "title": title,
            "wikipedia_page": page_title,
            "worldwide_box_office": self.parse_monetary_value(box_office_data['worldwide_box_office']),
            "domestic_box_office": self.parse_monetary_value(box_office_data['domestic_box_office']),
            "budget": self.parse_monetary_value(box_office_data['budget'])
        }
        
        return result

    def save_data(self, data, filename):
        """Save data to JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def run(self, movie_data_file="tmdb_movies.json"):
        """Main execution: fetch Wikipedia box office data for movies."""
        logger.info("Starting Wikipedia Box Office Data Collection...")
        
        # Load TMDB movies
        tmdb_file = os.path.join(config.RAW_DATA_DIR, movie_data_file)
        if not os.path.exists(tmdb_file):
            logger.error(f"TMDB data file not found: {tmdb_file}")
            return
        
        with open(tmdb_file, 'r') as f:
            tmdb_movies = json.load(f)
        
        box_office_data = {}
        
        for idx, movie in enumerate(tmdb_movies):
            title = movie.get('title', '')
            release_date = movie.get('release_date', '')
            movie_id = movie.get('id', idx)
            
            if not title:
                continue
            
            year = release_date.split('-')[0] if release_date else None
            
            # Fetch box office data
            box_office = self.fetch_movie_box_office(title, year)
            
            if box_office:
                box_office_data[str(movie_id)] = box_office
            
            # Rate limit: ~0.5 seconds between requests
            if idx % 5 == 0:
                logger.info(f"Processed {idx}/{len(tmdb_movies)} movies...")
        
        self.save_data(box_office_data, "wikipedia_box_office.json")
        logger.info(f"Completed! Collected box office data for {len(box_office_data)} movies.")


if __name__ == "__main__":
    collector = WikipediaBoxOfficeCollector()
    collector.run()
