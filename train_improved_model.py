import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_improved_regression():
    dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    df = pd.read_csv(dataset_path)
    
    logger.info(f"Dataset: {len(df)} movies")
    
    # ALL numeric features including sentiment
    numeric_features = [
        "budget", "runtime", "popularity", "vote_average", "vote_count",
        "trailer_views", "trailer_likes", "trailer_comments",
        "trailer_popularity_index", "interaction_rate", "engagement_velocity",
        "youtube_sentiment", "sentiment_volatility", "trend_momentum",
        "num_cast_members", "avg_cast_popularity", "max_cast_popularity", 
        "star_power_score", "avg_cast_historical_roi",
        "num_directors", "avg_director_popularity", "max_director_popularity", 
        "director_experience_score",
        "num_composers", "avg_composer_popularity", "max_composer_popularity", 
        "music_prestige_score",
        "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
        "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
        "release_month"
    ]
    
    # Check which exist
    available_features = [f for f in numeric_features if f in df.columns]
    logger.info(f"Available numeric features: {len(available_features)}")
    
    # Impute missing values with median
    X = df[available_features].copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle infinity values
    X = X.replace([np.inf, -np.inf], 0)
    
    y = df["revenue"]
    
    # Remove outlier samples (revenue > 3B is unrealistic)
    valid_idx = y < 3e9
    X = X[valid_idx]
    y = y[valid_idx]
    logger.info(f"After outlier removal: {len(X)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning for LightGBM
    logger.info("Training LightGBM with hyperparameter tuning...")
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], 
                  callbacks=[lgb.early_stopping(10)])
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred = lgb_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs(y_test.values - y_pred) / np.maximum(y_test.values, 1) * 100)
    
    errors = np.abs(y_test.values - y_pred) / np.maximum(y_test.values, 1) * 100
    under_40 = (errors < 40).sum() / len(errors) * 100
    
    logger.info(f"\nImproved LightGBM Results:")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  RMSE: ${rmse/1e6:.1f}M")
    logger.info(f"  MAE: ${mae/1e6:.1f}M")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info(f"  Movies with <40% error: {under_40:.1f}%")
    
    # Save with scaler
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "feature_scaler.pkl"))
    joblib.dump(lgb_model, os.path.join(config.MODELS_DIR, "best_regression_model.pkl"))
    logger.info("Model saved!")

if __name__ == "__main__":
    train_improved_regression()
