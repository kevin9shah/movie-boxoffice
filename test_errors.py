import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('data/processed/final_dataset.csv')
reg_model = joblib.load('models/best_regression_model.pkl')

# Get features that actually exist
features = [
    "budget", "runtime", "popularity", "vote_average", "vote_count",
    "trailer_views", "trailer_likes", "trailer_comments",
    "trailer_popularity_index", "interaction_rate", "engagement_velocity",
    "youtube_sentiment", "sentiment_volatility", "trend_momentum",
    "num_cast_members", "avg_cast_popularity", "max_cast_popularity", "star_power_score",
    "num_directors", "avg_director_popularity", "max_director_popularity", "director_experience_score",
    "num_composers", "avg_composer_popularity", "max_composer_popularity", "music_prestige_score",
    "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
    "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
    "overview", "primary_genre", "release_month"
]

available = [f for f in features if f in df.columns]
print(f"Using {len(available)} features")

# Filter to movies with revenue > 0
df_valid = df[df['revenue'] > 0].copy()
X = df_valid[available].fillna(0)
X['overview'] = X['overview'].fillna('').astype(str)

y_true = df_valid['revenue'].values
y_pred = reg_model.predict(X)

errors = np.abs(y_pred - y_true) / np.maximum(y_true, 1) * 100

print(f"Movies analyzed: {len(errors)}")
print(f"Mean error: {errors.mean():.1f}%")
print(f"Median error: {np.median(errors):.1f}%")
print(f"Movies >100% error: {(errors > 100).sum()}")
print(f"Movies >500% error: {(errors > 500).sum()}")
