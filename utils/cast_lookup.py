import re
import requests
import logging
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class CastLookup:
    def __init__(self):
        self.api_key = config.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json;charset=utf-8"
        }
        
    def find_person(self, name):
        """Search for a person by name on TMDB and return their details."""
        if not self.api_key:
            return None
            
        url = f"{self.base_url}/search/person"
        params = {
            "api_key": self.api_key,
            "query": name,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    # Return the most popular result
                    best_match = max(results, key=lambda x: x.get('popularity', 0))
                    return best_match
        except Exception as e:
            logger.warning(f"Error searching for person '{name}': {e}")
            
        return None

    def extract_names_from_text(self, text):
        """
        Extract potential cast/crew names from text using heuristics.
        Looks for patterns like "Actor Name - Role" or "Director: Name".
        """
        names = set()
        
        # Pattern 1: "Name – Role" or "Name - Role" (bullet points often used)
        # e.g. "Leonardo DiCaprio – returns as Dom Cobb"
        # We look for lines starting with a Name followed by a dash/hyphen
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Remove bullets
            line = re.sub(r'^[-•*]\s+', '', line)
            
            # Check for "Name - Role" pattern
            # Assuming names are 2-3 words, capitalized
            match = re.match(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\s*[-–—]', line)
            if match:
                names.add(match.group(1))
                continue
                
            # Check for "Role: Name, Name, Name" pattern
            # e.g. "Cast: Leonardo DiCaprio, Cillian Murphy"
            match_role = re.match(r'^(?:Director|Producer|Writer|Composer|Cinematography|Music|Starring|Cast|Main Cast|Crew):\s*(.*)', line, re.IGNORECASE)
            if match_role:
                content = match_role.group(1)
                # Split by commas or specific separators
                potential_names = re.split(r'[,&]|\s+and\s+', content)
                for name in potential_names:
                    name = name.strip()
                    # Basic validation: 2-3 words, distinct casing
                    if re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}$', name):
                        names.add(name)
                        
        return list(names)

    def get_cast_popularity(self, text):
        """
        Parse text for names, lookup popularity, and return stats.
        Returns dict with avg_popularity, distinct_names_found, etc.
        """
        names = self.extract_names_from_text(text)
        if not names:
            return None
            
        logger.info(f"Dynamic Cast Lookup found names: {names}")
        
        popularities = []
        found_names = []
        
        for name in names:
            person = self.find_person(name)
            if person:
                pop = person.get('popularity', 0)
                popularities.append(pop)
                found_names.append(f"{name} ({pop:.1f})")
            else:
                logger.info(f"Could not find likely match for '{name}'")
        
        if not popularities:
            return None
            
        stats = {
            "avg_cast_popularity": sum(popularities) / len(popularities),
            "max_cast_popularity": max(popularities),
            "num_cast_members": len(popularities), # Only counted found ones
            "found_names": found_names
        }
        
        # Calculate Star Power
        stats["star_power_score"] = stats["avg_cast_popularity"] * stats["num_cast_members"]
        
        return stats
