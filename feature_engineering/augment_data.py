import pandas as pd
import numpy as np
import os
import sys
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_synthetic_flops(n_samples=100):
    """
    Generates synthetic 'flop' and 'bad' movie data to balance the dataset 
    and teach the model what a failure looks like.
    """
    
    print(f"Generating {n_samples} synthetic flop samples...")
    
    data = []
    
    # Keywords for bad movie descriptions
    bad_keywords = [
        "boring", "terrible", "waste", "disaster", "awful", "dull", "lifeless", 
        "uninspired", "mess", "failure", "flop", "worst", "garbage", "cliche",
        "predictable", "poorly", "amateur", "forgetful", "bland", "cheap",
        "pointless", "slow", "confusing", "weak", "badly", "horrible"
    ]
    
    genres = ["Action", "Comedy", "Drama", "Horror", "Thriller", "Science Fiction", "Romance"]
    
    for i in range(n_samples):
        # 1. Low Budget / Amateur
        # 2. High Budget / Box Office Bomb
        if random.random() < 0.7:
            # Low budget indie flop
            budget = random.randint(10000, 1000000)
            revenue = budget * random.uniform(0.1, 0.8) # Loss
        else:
            # High budget bomb
            budget = random.randint(20000000, 100000000)
            revenue = budget * random.uniform(0.1, 0.5) # Huge Loss
            
        runtime = random.randint(60, 100) # Often shorter or oddly long, but let's say short/avg
        
        # Popularity & Socials - VERY LOW
        popularity = random.uniform(0.1, 5.0)
        vote_average = random.uniform(1.0, 4.5)
        vote_count = random.randint(5, 50)
        
        # Trailer stats - minimal engagement
        trailer_views = int(popularity * 1000)
        trailer_likes = int(trailer_views * 0.01)
        trailer_comments = int(trailer_likes * 0.1)
        trailer_pop_index = 0.1
        
        # Social Sentiment - Negative
        youtube_sentiment = random.uniform(0.0, 3.0)
        sentiment_volatility = random.uniform(0.1, 0.3)
        interaction_rate = 0.01
        engagement_velocity = 0.0
        trend_momentum = 0.0
        
        # Cast & Crew - Unknowns
        num_cast = random.randint(1, 15)
        avg_cast_pop = random.uniform(0.0, 5.0)
        max_cast_pop = avg_cast_pop * 1.2
        star_power = avg_cast_pop * num_cast
        
        num_directors = 1
        avg_dir_pop = random.uniform(0.0, 2.0)
        max_dir_pop = avg_dir_pop
        dir_exp = avg_dir_pop
        
        num_composers = 1
        avg_comp_pop = 0.0
        max_comp_pop = 0.0
        music_prestige = 0.0
        
        # Text Description
        # Construct a "bad" description
        desc_start = ["A", "The", "This"]
        desc_mid = ["movie is", "film is", "story is", "plot is"]
        adj = random.choice(bad_keywords)
        adj2 = random.choice(bad_keywords)
        
        description = f"{random.choice(desc_start)} {random.choice(desc_mid)} {adj} and {adj2}. It fails to deliver."
        
        sample = {
            "movie_id": f"synthetic_flop_{i}",
            "title": f"Bad Movie {i}",
            "budget": budget,
            "revenue": revenue,
            "runtime": runtime,
            "popularity": popularity,
            "vote_average": vote_average,
            "vote_count": vote_count,
            "release_date": "2020-01-01",
            "primary_genre": random.choice(genres),
            "trailer_views": trailer_views,
            "trailer_likes": trailer_likes,
            "trailer_comments": trailer_comments,
            "trailer_popularity_index": trailer_pop_index,
            "interaction_rate": interaction_rate,
            "engagement_velocity": engagement_velocity,
            "release_month": random.randint(1, 12),
            "youtube_sentiment": youtube_sentiment,
            "sentiment_volatility": sentiment_volatility,
            "trend_momentum": trend_momentum,
            "num_cast_members": num_cast,
            "avg_cast_popularity": avg_cast_pop,
            "max_cast_popularity": max_cast_pop,
            "star_power_score": star_power,
            "num_directors": num_directors,
            "avg_director_popularity": avg_dir_pop,
            "max_director_popularity": max_dir_pop,
            "director_experience_score": dir_exp,
            "num_composers": num_composers,
            "avg_composer_popularity": avg_comp_pop,
            "max_composer_popularity": max_comp_pop,
            "music_prestige_score": music_prestige,
            "wikipedia_worldwide_box_office": 0.0,
            "wikipedia_budget": 0.0,
            "overview": description,
            "success_class": "Flop"
        }
        
        data.append(sample)
        
    return pd.DataFrame(data)

def augment_dataframe():
    """Load existing dataset, augment, and save."""
    dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        existing_df = pd.read_csv(dataset_path)
        
        # Filter out previous synthetic data if any
        existing_df = existing_df[~existing_df['movie_id'].astype(str).str.startswith('synthetic_flop')]
        
        print(f"Existing samples: {len(existing_df)}")
    else:
        print("Existing dataset not found, starting fresh (unlikely for augmentation task).")
        return

    # Generate Synthetic Data
    synthetic_df = generate_synthetic_flops(n_samples=100)
    
    # Combine
    augmented_df = pd.concat([existing_df, synthetic_df], ignore_index=True)
    
    # Save
    augmented_df.to_csv(dataset_path, index=False)
    print(f"Augmented dataset saved. New shape: {augmented_df.shape}")
    
    # Verify Distribution
    print("\nNew Class Distribution:")
    print(augmented_df['success_class'].value_counts())

if __name__ == "__main__":
    augment_dataframe()
