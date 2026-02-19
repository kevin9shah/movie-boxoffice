import pandas as pd
import json
import os
import sys
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CastCrewFeatures:
    def __init__(self):
        self.cast_crew_data = None
        self.wikipedia_box_office = None

    def load_cast_crew_data(self, filename="tmdb_cast_crew.json"):
        """Load cast and crew data from JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Cast/crew data file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                self.cast_crew_data = json.load(f)
            logger.info(f"Loaded cast/crew data from {filepath}")
            return self.cast_crew_data
        except Exception as e:
            logger.error(f"Error loading cast/crew data: {e}")
            return {}

    def load_wikipedia_box_office(self, filename="wikipedia_box_office.json"):
        """Load Wikipedia box office data."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Wikipedia box office data file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                self.wikipedia_box_office = json.load(f)
            logger.info(f"Loaded Wikipedia box office data from {filepath}")
            return self.wikipedia_box_office
        except Exception as e:
            logger.error(f"Error loading Wikipedia box office data: {e}")
            return {}

    def extract_features_from_cast(self, cast_list):
        """Extract features from cast list."""
        features = {
            "num_cast_members": len(cast_list),
            "avg_cast_popularity": 0,
            "max_cast_popularity": 0,
            "star_power_score": 0
        }
        
        if not cast_list:
            return features
        
        popularities = [actor.get('popularity', 0) for actor in cast_list]
        
        if popularities:
            features['avg_cast_popularity'] = sum(popularities) / len(popularities)
            features['max_cast_popularity'] = max(popularities)
            # Star power: average popularity * number of cast members (within top 5)
            features['star_power_score'] = features['avg_cast_popularity'] * len(cast_list)
        
        return features

    def extract_features_from_directors(self, director_list):
        """Extract features from director list."""
        features = {
            "num_directors": len(director_list),
            "avg_director_popularity": 0,
            "max_director_popularity": 0,
            "director_experience_score": 0
        }
        
        if not director_list:
            return features
        
        popularities = [d.get('popularity', 0) for d in director_list]
        
        if popularities:
            features['avg_director_popularity'] = sum(popularities) / len(popularities)
            features['max_director_popularity'] = max(popularities)
            # Experience score: how established the director is
            features['director_experience_score'] = features['avg_director_popularity']
        
        return features

    def extract_features_from_composers(self, composer_list):
        """Extract features from music composer list."""
        features = {
            "num_composers": len(composer_list),
            "avg_composer_popularity": 0,
            "max_composer_popularity": 0,
            "music_prestige_score": 0
        }
        
        if not composer_list:
            return features
        
        popularities = [c.get('popularity', 0) for c in composer_list]
        
        if popularities:
            features['avg_composer_popularity'] = sum(popularities) / len(popularities)
            features['max_composer_popularity'] = max(popularities)
            # Prestige: how famous the composer is
            features['music_prestige_score'] = features['avg_composer_popularity']
        
        return features

    def generate_cast_crew_features(self, df, cast_crew_data=None):
        """Generate cast and crew features for a dataframe."""
        if cast_crew_data is None:
            cast_crew_data = self.load_cast_crew_data()
        
        if not cast_crew_data:
            logger.warning("No cast/crew data available, returning empty features")
            return df
        
        # Initialize new columns
        cast_features = ['num_cast_members', 'avg_cast_popularity', 'max_cast_popularity', 'star_power_score']
        director_features = ['num_directors', 'avg_director_popularity', 'max_director_popularity', 'director_experience_score']
        composer_features = ['num_composers', 'avg_composer_popularity', 'max_composer_popularity', 'music_prestige_score']
        
        all_cast_features = cast_features + director_features + composer_features
        
        for feature in all_cast_features:
            df[feature] = 0.0
        
        # Extract features for each movie
        for idx, row in df.iterrows():
            movie_id = str(row.get('id', idx))
            
            if movie_id in cast_crew_data:
                cast_crew = cast_crew_data[movie_id]
                
                # Cast features
                if 'cast' in cast_crew:
                    cast_feats = self.extract_features_from_cast(cast_crew['cast'])
                    for feature in cast_features:
                        if feature in cast_feats:
                            df.at[idx, feature] = cast_feats[feature]
                
                # Director features
                if 'directors' in cast_crew:
                    director_feats = self.extract_features_from_directors(cast_crew['directors'])
                    for feature in director_features:
                        if feature in director_feats:
                            df.at[idx, feature] = director_feats[feature]
                
                # Composer features
                if 'composers' in cast_crew:
                    composer_feats = self.extract_features_from_composers(cast_crew['composers'])
                    for feature in composer_features:
                        if feature in composer_feats:
                            df.at[idx, feature] = composer_feats[feature]
        
        logger.info(f"Added {len(all_cast_features)} cast/crew features to dataframe")
        return df

    def generate_wikipedia_box_office_features(self, df, wikipedia_data=None):
        """Generate features from Wikipedia box office data."""
        if wikipedia_data is None:
            wikipedia_data = self.load_wikipedia_box_office()
        
        if not wikipedia_data:
            logger.warning("No Wikipedia box office data available")
            # Initialize columns with nulls
            df['wikipedia_worldwide_box_office'] = None
            df['wikipedia_budget'] = None
            return df
        
        # Initialize new columns
        df['wikipedia_worldwide_box_office'] = None
        df['wikipedia_budget'] = None
        
        # Extract features for each movie
        for idx, row in df.iterrows():
            movie_id = str(row.get('id', idx))
            
            if movie_id in wikipedia_data:
                wiki_data = wikipedia_data[movie_id]
                df.at[idx, 'wikipedia_worldwide_box_office'] = wiki_data.get('worldwide_box_office')
                df.at[idx, 'wikipedia_budget'] = wiki_data.get('budget')
        
        logger.info("Added Wikipedia box office features to dataframe")
        return df

    def generate_all_features(self, df):
        """Generate all cast, crew, and Wikipedia features."""
        logger.info("Generating cast, crew, and Wikipedia features...")
        
        # Load data
        cast_crew_data = self.load_cast_crew_data()
        wikipedia_data = self.load_wikipedia_box_office()
        
        # Generate features
        df = self.generate_cast_crew_features(df, cast_crew_data)
        df = self.generate_wikipedia_box_office_features(df, wikipedia_data)
        
        logger.info("Completed generating all cast/crew/Wikipedia features")
        return df


def main():
    """Test the cast crew features module."""
    logger.info("Testing Cast/Crew Features Module...")
    
    # Create a sample dataframe
    sample_data = {
        'id': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3']
    }
    df = pd.DataFrame(sample_data)
    
    # Generate features
    feature_gen = CastCrewFeatures()
    df = feature_gen.generate_all_features(df)
    
    logger.info("\nFeature columns added:")
    logger.info(df.columns.tolist())
    logger.info(f"\nDataframe shape: {df.shape}")
    logger.info(f"\nFirst few rows:\n{df.head()}")


if __name__ == "__main__":
    main()
