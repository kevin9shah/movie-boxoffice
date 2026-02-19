import pandas as pd
import os
import sys
import json
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .cast_crew_features import CastCrewFeatures

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def integrate_features():
    logger.info("Starting Feature Integration...")
    
    # 1. Load Spark Features (Parquet)
    spark_features_path = os.path.join(config.PROCESSED_DATA_DIR, "spark_features")
    if not os.path.exists(spark_features_path):
        logger.error("Spark features not found.")
        return
        
    try:
        df = pd.read_parquet(spark_features_path)
    except Exception as e:
        logger.error(f"Error reading Parquet: {e}")
        return

    # 2. Load Sentiment Data (JSON)
    # YouTube Sentiment
    yt_sent_path = os.path.join(config.PROCESSED_DATA_DIR, "youtube_sentiment.json")
    if os.path.exists(yt_sent_path):
        with open(yt_sent_path, 'r') as f:
            yt_sent_data = json.load(f)
        yt_sent_df = pd.DataFrame(yt_sent_data)
        if not yt_sent_df.empty:
            df = df.merge(yt_sent_df, on="movie_id", how="left")
            
    # 3. Load Google Trends (JSON)
    trends_path = os.path.join(config.RAW_DATA_DIR, "google_trends.json")
    if os.path.exists(trends_path):
        with open(trends_path, 'r') as f:
            trends_data = json.load(f)
        
        # Calculate trend momentum (slope or avg)
        trend_features = []
        for title, timeline in trends_data.items():
            if not timeline:
                 trend_features.append({"title": title, "trend_momentum": 0})
                 continue
                 
            # Simple avg of last 5 points
            # Ensure timeline values are numeric
            values = [float(v) for v in timeline.values() if str(v).replace('.', '', 1).isdigit()]
            if values:
                avg_trend = sum(values) / len(values)
            else:
                 avg_trend = 0
            
            trend_features.append({"title": title, "trend_momentum": avg_trend})
            
        trends_df = pd.DataFrame(trend_features)
        
        if not trends_df.empty:
            df['title_lower'] = df['title'].str.lower()
            trends_df['title_lower'] = trends_df['title'].str.lower()
            df = df.merge(trends_df[['title_lower', 'trend_momentum']], 
                          on="title_lower", how="left")
            df.drop(columns=['title_lower'], inplace=True)

    # 4. ADD NEW: Cast, Crew, and Wikipedia Box Office Features
    logger.info("Adding Cast, Crew, and Wikipedia Box Office Features...")
    cast_crew_gen = CastCrewFeatures()
    df = cast_crew_gen.generate_all_features(df)

    # 4b. ADD NEW: Movie Description (Overview)
    logger.info("Adding Movie Descriptions (Overview)...")
    tmdb_path = os.path.join(config.RAW_DATA_DIR, "tmdb_movies.json")
    if os.path.exists(tmdb_path):
        with open(tmdb_path, 'r') as f:
            tmdb_data = json.load(f)
        
        # Create a mapping of id -> overview
        overview_map = {m['id']: m.get('overview', '') for m in tmdb_data}
        df['overview'] = df['movie_id'].map(overview_map).fillna('')
    else:
        logger.warning("TMDB data not found, overview will be empty.")
        df['overview'] = ""
    
    # 5. Fill NaNs
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Handle categorical columns
    if 'primary_genre' in df.columns:
        df['primary_genre'] = df['primary_genre'].fillna("Unknown")
    
    # Ensure release_month is int
    if 'release_month' in df.columns:
        df['release_month'] = df['release_month'].astype(int)
    
    # 6. Create Target Variables (if not present or for classification)
    # Revenue is the target, 'revenue' column exists from TMDB
    # Classification: Hit > 3x Budget, Average > 1x Budget, Flop < 1x Budget
    def classify_success(row):
        if row['budget'] == 0:
            return "Unknown"
        ratio = row['revenue'] / row['budget']
        if ratio >= 3:
            return "Hit"
        elif ratio >= 1:
            return "Average"
        else:
            return "Flop"

    df['success_class'] = df.apply(classify_success, axis=1)

    # Deduplicate based on movie_id
    logger.info(f"Shape before deduplication: {df.shape}")
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    logger.info(f"Shape after deduplication: {df.shape}")

    # Save Final Dataset
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Final dataset saved to {output_path}. Shape: {df.shape}")
    logger.info(f"Columns in final dataset: {list(df.columns)}")

if __name__ == "__main__":
    integrate_features()
