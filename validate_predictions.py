import pandas as pd
import os
import sys
import logging
import joblib

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_predictions():
    logger.info("Starting Prediction Validation...")
    
    # Load processed data
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    if not os.path.exists(data_path):
        logger.error("Processed data not found.")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} movies for validation.")
    
    # Load Regression Model
    model_path = os.path.join(config.MODELS_DIR, "best_regression_model.pkl")
    if not os.path.exists(model_path):
        logger.error("Regression model not found.")
        return
        
    try:
        reg_model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Define features (Must match training)
    features = [
        "budget", "runtime", "popularity", "vote_average", "vote_count",
        "trailer_views", "trailer_likes", "trailer_comments",
        "trailer_popularity_index", "interaction_rate", "engagement_velocity",
        "youtube_sentiment", "sentiment_volatility",
        "trend_momentum",
        "primary_genre", "release_month"
    ]
    
    # Filter for movies with known revenue (revenue > 1000)
    valid_df = df[df['revenue'] > 1000].copy()
    valid_df = valid_df.fillna(0) # Fill NaNs (important for new cols)
    
    if valid_df.empty:
        logger.warning("No movies with valid known revenue found in the dataset.")
    else:
        logger.info(f"Validating against {len(valid_df)} movies with known revenue.")
        
        X_val = valid_df[features].fillna(0)
        
        # Predict
        valid_df['predicted_revenue'] = reg_model.predict(X_val)
        valid_df['difference'] = valid_df['predicted_revenue'] - valid_df['revenue']
        valid_df['error_pct'] = (valid_df['difference'].abs() / valid_df['revenue']) * 100
        
        # Display Results
        print("\n--- Validation Results (Top 20 by Revenue) ---")
        print(valid_df[['title', 'revenue', 'predicted_revenue', 'error_pct']].sort_values('revenue', ascending=False).head(20).to_string(index=False))
        
        # Calculate Metrics
        mean_error_pct = valid_df['error_pct'].mean()
        logger.info(f"\nMean Absolute Percentage Error (MAPE): {mean_error_pct:.2f}%")

    # Also show predictions for unreleased/unknown revenue movies (Bottom 10)
    unknown_df = df[df['revenue'] <= 1000].copy()
    if not unknown_df.empty:
        X_unknown = unknown_df[features].fillna(0)
        unknown_df['predicted_revenue'] = reg_model.predict(X_unknown)
        
        print("\n--- Predictions for Movies with Unknown Revenue (Top 10 by Prediction) ---")
        print(unknown_df[['title', 'predicted_revenue']].sort_values('predicted_revenue', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    validate_predictions()
