import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append('.')
import config

# Load data and model
df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'final_dataset.csv'))
reg_model = joblib.load(os.path.join(config.MODELS_DIR, 'best_regression_model.pkl'))

# Prepare features
features = [
    "budget", "runtime", "popularity", "vote_average", "vote_count",
    "trailer_views", "trailer_likes", "trailer_comments",
    "trailer_popularity_index", "interaction_rate", "engagement_velocity",
    "youtube_sentiment", "sentiment_volatility", "trend_momentum",
    "num_cast_members", "avg_cast_popularity", "max_cast_popularity", "star_power_score", "avg_cast_historical_roi",
    "num_directors", "avg_director_popularity", "max_director_popularity", "director_experience_score",
    "num_composers", "avg_composer_popularity", "max_composer_popularity", "music_prestige_score",
    "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
    "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
    "overview", "primary_genre", "release_month"
]

available = [f for f in features if f in df.columns]
X = df[available].fillna(0).copy()
X['overview'] = X['overview'].fillna('').astype(str)

y_true = df['revenue'].values
y_pred = reg_model.predict(X)

# Calculate errors
errors = np.abs(y_pred - y_true) / np.maximum(y_true, 1) * 100
df['pred_error'] = errors

print("="*70)
print("ERROR ANALYSIS - ROOT CAUSES")
print("="*70)
print(f"\nError statistics:")
print(f"  Mean error: {errors.mean():.1f}%")
print(f"  Median: {np.median(errors):.1f}%")
print(f"  > 40% error: {(errors >= 40).sum()} movies ({(errors >= 40).sum()/len(errors)*100:.1f}%)")

print(f"\nError by genre (avg):")
genre_err = df.groupby('primary_genre')['pred_error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(genre_err)

print(f"\nMovies with 40%+ errors (by genre):")
high_err = df[df['pred_error'] >= 40].groupby('primary_genre').size().sort_values(ascending=False)
print(high_err)

print(f"\nFeature analysis for high-error movies:")
high_error_df = df[df['pred_error'] >= 40]
print(f"  Avg popularity: {high_error_df['popularity'].mean():.1f} vs overall {df['popularity'].mean():.1f}")
print(f"  Avg vote_count: {high_error_df['vote_count'].mean():.0f} vs overall {df['vote_count'].mean():.0f}")
print(f"  Zero trailer_views: {(high_error_df['trailer_views']==0).sum()} out of {len(high_error_df)}")

print("="*70)
