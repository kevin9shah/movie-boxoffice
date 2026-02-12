from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_spark_session():
    return SparkSession.builder \
        .appName("MovieFeatureEngineering") \
        .master("local[*]") \
        .getOrCreate()

def run_feature_engineering():
    spark = create_spark_session()
    
    # Load Cleaned Data
    tmdb_path = os.path.join(config.PROCESSED_DATA_DIR, "tmdb_cleaned")
    youtube_path = os.path.join(config.PROCESSED_DATA_DIR, "youtube_cleaned")
    
    if not os.path.exists(tmdb_path) or not os.path.exists(youtube_path):
        print("Cleaned data not found. Run spark_cleaning.py first.")
        return

    tmdb_df = spark.read.parquet(tmdb_path)
    yt_df = spark.read.parquet(youtube_path)

    # Join TMDB and YouTube
    # Assuming 1:1 mapping for simplicity here
    combined_df = tmdb_df.join(yt_df, on="movie_id", how="left")
    
    # Feature Engineering
    
    # 1. Trailer Popularity Index (normalized views)
    # Simple normalization for demo: views / 1M
    combined_df = combined_df.withColumn(
        "trailer_popularity_index", 
        col("trailer_views") / 1000000
    )
    
    # 2. Interaction Rate (likes / views)
    combined_df = combined_df.withColumn(
        "interaction_rate",
        when(col("trailer_views") > 0, col("trailer_likes") / col("trailer_views")).otherwise(0)
    )
    
    # 3. Engagement Velocity (views / days since upload - mocked as simplistic here)
    # real dataset would need upload date. We'll use a proxy or placehold.
    combined_df = combined_df.withColumn(
        "engagement_velocity",
        col("trailer_views") / 30 # average views per day over a month approx
    )

    # 4. Release Seasonality (Month)
    # Extract month from release_date string (YYYY-MM-DD)
    combined_df = combined_df.withColumn(
        "release_month",
        when(col("release_date").isNotNull(), 
             col("release_date").substr(6, 2).cast("int"))
        .otherwise(0)
    )

    # Save Intermediate Features
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "spark_features")
    combined_df.write.mode("overwrite").parquet(output_path)
    print(f"Saved Spark features to {output_path}")
    
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()
