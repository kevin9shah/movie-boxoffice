from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, current_timestamp
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_spark_session():
    return SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master("local[*]") \
        .getOrCreate()

def clean_tmdb_data(spark):
    """Clean TMDB data."""
    tmdb_path = os.path.join(config.RAW_DATA_DIR, "tmdb_movies.json")
    if not os.path.exists(tmdb_path):
        print(f"File not found: {tmdb_path}")
        return None

    df = spark.read.option("multiline", "true").json(tmdb_path)
    
    # Select relevant columns and handle nulls
    # Select relevant columns and handle nulls
    # Extract primary genre from array of struct
    cleaned_df = df.select(
        col("id").alias("movie_id"),
        col("title"),
        col("budget"),
        col("revenue"),
        col("runtime"),
        col("popularity"),
        col("vote_average"),
        col("vote_count"),
        col("release_date"),
        col("genres")[0]["name"].alias("primary_genre")
    ).na.fill({
        "budget": 0,
        "revenue": 0,
        "popularity": 0.0,
        "primary_genre": "Unknown"
    })
    
    return cleaned_df

def clean_youtube_data(spark):
    """Clean YouTube data."""
    yt_path = os.path.join(config.RAW_DATA_DIR, "youtube_trailers.json")
    if not os.path.exists(yt_path):
        print(f"File not found: {yt_path}")
        return None
        
    df = spark.read.option("multiline", "true").json(yt_path)
    
    # Extract stats from struct
    cleaned_df = df.select(
        col("movie_id"),
        col("trailer_stats.viewCount").cast("long").alias("trailer_views"),
        col("trailer_stats.likeCount").cast("long").alias("trailer_likes"),
        col("trailer_stats.commentCount").cast("long").alias("trailer_comments")
    ).na.fill(0)
    
    return cleaned_df

def main():
    spark = create_spark_session()
    
    # Process TMDB
    tmdb_df = clean_tmdb_data(spark)
    if tmdb_df:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "tmdb_cleaned")
        tmdb_df.write.mode("overwrite").parquet(output_path)
        print(f"Saved cleaned TMDB data to {output_path}")

    # Process YouTube
    yt_df = clean_youtube_data(spark)
    if yt_df:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "youtube_cleaned")
        yt_df.write.mode("overwrite").parquet(output_path)
        print(f"Saved cleaned YouTube data to {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
