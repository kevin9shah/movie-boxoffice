"""
Complete pipeline: Enrich cast/crew data from TMDB + Retrain model.
Removes wikipedia_worldwide_box_office (data leakage).
Uses sample weights to prioritize blockbuster accuracy.
"""
import pandas as pd
import numpy as np
import requests
import os, sys, time, json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

TMDB_KEY = "14acdea37a4e7f5bb4c2be5592ee182c"
BASE_URL = "https://api.themoviedb.org/3"
CACHE_FILE = os.path.join(config.RAW_DATA_DIR, "credits_cache.json")

# Load or create credits cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        credits_cache = json.load(f)
    logger.info(f"Loaded {len(credits_cache)} cached credits")
else:
    credits_cache = {}

def fetch_credits(movie_id):
    """Fetch cast/crew credits from TMDB with caching."""
    movie_id_str = str(movie_id)
    # Skip synthetic/non-numeric IDs
    try:
        movie_id_str = str(int(float(movie_id_str)))
    except (ValueError, TypeError):
        return None
    if movie_id_str in credits_cache:
        return credits_cache[movie_id_str]
    
    try:
        r = requests.get(f"{BASE_URL}/movie/{movie_id_str}/credits",
                        params={"api_key": TMDB_KEY}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Extract top 10 cast and all directors/composers
            cast = data.get("cast", [])[:10]
            crew = data.get("crew", [])
            
            result = {
                "cast": [{"name": c.get("name", ""), "popularity": c.get("popularity", 0)} for c in cast],
                "directors": [{"name": c.get("name", ""), "popularity": c.get("popularity", 0)} 
                             for c in crew if c.get("job") == "Director"],
                "composers": [{"name": c.get("name", ""), "popularity": c.get("popularity", 0)} 
                             for c in crew if c.get("job") in ("Original Music Composer", "Music")]
            }
            credits_cache[movie_id_str] = result
            return result
        elif r.status_code == 429:
            time.sleep(2)
            return fetch_credits(movie_id)
    except Exception as e:
        logger.error(f"Error fetching credits for {movie_id}: {e}")
    
    return None

def compute_cast_features(credits):
    """Compute cast/crew features from credits data."""
    if not credits:
        return {}
    
    cast = credits.get("cast", [])
    directors = credits.get("directors", [])
    composers = credits.get("composers", [])
    
    cast_pops = [c["popularity"] for c in cast if c["popularity"] > 0]
    dir_pops = [d["popularity"] for d in directors if d["popularity"] > 0]
    comp_pops = [c["popularity"] for c in composers if c["popularity"] > 0]
    
    features = {
        "num_cast_members": len(cast),
        "avg_cast_popularity": np.mean(cast_pops) if cast_pops else 0,
        "max_cast_popularity": max(cast_pops) if cast_pops else 0,
        "star_power_score": sum(sorted(cast_pops, reverse=True)[:3]) if cast_pops else 0,
        "num_directors": len(directors),
        "avg_director_popularity": np.mean(dir_pops) if dir_pops else 0,
        "max_director_popularity": max(dir_pops) if dir_pops else 0,
        "director_experience_score": sum(dir_pops) if dir_pops else 0,
        "num_composers": len(composers),
        "avg_composer_popularity": np.mean(comp_pops) if comp_pops else 0,
        "max_composer_popularity": max(comp_pops) if comp_pops else 0,
        "music_prestige_score": sum(comp_pops) if comp_pops else 0,
    }
    return features

# ===== STEP 1: ENRICH CAST/CREW =====
logger.info("=" * 60)
logger.info("STEP 1: ENRICHING CAST/CREW DATA FROM TMDB")
logger.info("=" * 60)

df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv"))
df = df.fillna(0)

# Convert cast/crew columns to float to avoid dtype errors
float_cols = ['num_cast_members', 'avg_cast_popularity', 'max_cast_popularity', 'star_power_score',
              'num_directors', 'avg_director_popularity', 'max_director_popularity', 'director_experience_score',
              'num_composers', 'avg_composer_popularity', 'max_composer_popularity', 'music_prestige_score']
for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype(float)
total = len(df)
enriched = 0

for idx, row in df.iterrows():
    movie_id = row["movie_id"]
    credits = fetch_credits(movie_id)
    
    if credits:
        features = compute_cast_features(credits)
        for key, value in features.items():
            df.loc[idx, key] = value
        enriched += 1
    
    if (idx + 1) % 50 == 0:
        logger.info(f"  Progress: {idx + 1}/{total} ({enriched} enriched)")
        # Save cache periodically
        with open(CACHE_FILE, 'w') as f:
            json.dump(credits_cache, f)
    
    time.sleep(0.25)  # Rate limit: ~4 req/sec

# Final cache save
with open(CACHE_FILE, 'w') as f:
    json.dump(credits_cache, f)

logger.info(f"  Enriched {enriched}/{total} movies with cast/crew data")

# Fill missing overviews
for idx in df.index:
    if pd.isna(df.loc[idx, 'overview']) or str(df.loc[idx, 'overview']).strip() in ['', '0']:
        genre = str(df.loc[idx, 'primary_genre'])
        df.loc[idx, 'overview'] = f"A {genre.lower()} film."

# Recompute engineered features that depend on enriched data
df['hype_score'] = (
    df['popularity'] * 0.3 + 
    df['vote_count'] * 0.001 + 
    df['trailer_views'] * 0.00001 +
    df['trailer_likes'] * 0.001
)

# Save enriched dataset
df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv"), index=False)
logger.info(f"  Saved enriched dataset ({len(df)} movies)")

# ===== STEP 2: RETRAIN MODEL =====
logger.info("=" * 60)
logger.info("STEP 2: RETRAINING MODEL (NO WIKIPEDIA LEAKAGE)")
logger.info("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Features WITHOUT wikipedia leakage
features = [
    "budget", "popularity", "vote_average", "vote_count",
    "trailer_views", "trailer_likes", "trailer_comments",
    "trailer_popularity_index", "interaction_rate", "engagement_velocity",
    "youtube_sentiment", "sentiment_volatility", "trend_momentum",
    # Cast/crew (NOW ENRICHED!)
    "num_cast_members", "avg_cast_popularity", "max_cast_popularity", "star_power_score",
    "num_directors", "avg_director_popularity", "max_director_popularity", "director_experience_score",
    "num_composers", "avg_composer_popularity", "max_composer_popularity", "music_prestige_score",
    # Engineered features
    "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
    "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
    # Text + categorical
    "overview", "primary_genre", "release_month"
]

available = [f for f in features if f in df.columns]
logger.info(f"  Using {len(available)} features (no wikipedia leakage)")

X = df[available].copy()
X['overview'] = X['overview'].fillna('').astype(str)
y = df['revenue']

# Sample weights: blockbusters matter more
sample_weights = np.log1p(y / y.median())
sample_weights = sample_weights / sample_weights.mean()

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)
logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

categorical_features = ['primary_genre', 'release_month']
text_features = ['overview']
numeric_features = [c for c in X.columns if c not in categorical_features and c not in text_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('txt', TfidfVectorizer(max_features=200, stop_words='english'), 'overview')
    ])

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=10, 
                                 learning_rate=0.08, subsample=0.8, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=12, learning_rate=0.08, 
                                   num_leaves=50, random_state=42, verbose=-1),
}

best_model = None
best_name = ""
best_r2 = -float('inf')

for name, model in models.items():
    logger.info(f"  Training {name}...")
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    clf.fit(X_train, y_train, regressor__sample_weight=w_train)
    
    y_pred = clf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mape_values = np.abs(y_test - y_pred) / np.maximum(y_test, 1) * 100
    avg_mape = np.mean(mape_values)
    
    logger.info(f"    R²={r2:.4f}, MAE=${mae/1e6:.1f}M, MAPE={avg_mape:.1f}%")
    
    if r2 > best_r2:
        best_r2 = r2
        best_name = name
        best_model = clf

logger.info(f"\n  BEST: {best_name} (R²={best_r2:.4f})")
joblib.dump(best_model, os.path.join(config.MODELS_DIR, "best_regression_model.pkl"))
logger.info(f"  Saved best_regression_model.pkl")

# ===== STEP 3: QUICK VALIDATION =====
logger.info("=" * 60)
logger.info("STEP 3: QUICK VALIDATION (Top 20 by Revenue)")
logger.info("=" * 60)

top20 = df.nlargest(20, 'revenue')
errors = []
for _, row in top20.iterrows():
    x_row = pd.DataFrame([row[available]])
    x_row['overview'] = x_row['overview'].fillna('').astype(str)
    pred = best_model.predict(x_row)[0]
    actual = row['revenue']
    err = abs(pred - actual) / actual * 100
    errors.append(err)
    logger.info(f"  {row['title']:<40} Actual=${actual/1e6:>8.0f}M  Pred=${pred/1e6:>8.0f}M  Err={err:>5.1f}%")

logger.info(f"\n  Avg Error (Top 20): {np.mean(errors):.1f}%")
logger.info("DONE!")
