#!/usr/bin/env python3

import joblib
import pandas as pd
import sys
sys.path.append('.')
from api.app import MovieInput, calculate_revenue

# Load model
reg_model = joblib.load('models/best_regression_model.pkl')

# Create a test movie input
movie = MovieInput(
    title="Test Movie",
    budget=100000000,
    runtime=120,
    popularity=50,
    vote_average=7.5,
    vote_count=5000,
    trailer_views=5000000,
    trailer_likes=50000,
    trailer_comments=5000,
    trailer_popularity_index=0.5,
    interaction_rate=0.05,
    engagement_velocity=0.1,
    youtube_sentiment=7.0,
    sentiment_volatility=0.2,
    trend_momentum=0.1,
    num_cast_members=20,
    avg_cast_popularity=30,
    max_cast_popularity=80,
    star_power_score=70,
    num_directors=1,
    avg_director_popularity=40,
    max_director_popularity=70,
    director_experience_score=80,
    num_composers=1,
    avg_composer_popularity=30,
    max_composer_popularity=60,
    music_prestige_score=60,
    overview="A great action movie",
    primary_genre="Action",
    release_month=6,
    is_franchise=1,
    is_sequel=0,
    budget_tier=4,
    genre_avg_revenue=450000000,
    description_length=19,
    hype_score=150,
    budget_popularity_ratio=2000000,
    vote_power=37500
)

try:
    pred = calculate_revenue(movie, reg_model)
    print(f"Prediction: ${pred:,.0f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
