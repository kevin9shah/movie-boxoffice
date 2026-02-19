import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import sys
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RegressionModel:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""

    def load_data(self):
        dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
        if not os.path.exists(dataset_path):
            logger.error("Dataset not found.")
            return None, None
            
        df = pd.read_csv(dataset_path)
        
        # Select Features (including new cast, crew, Wikipedia, and DESCRIPTION features)
        features = [
            "budget", "popularity", "vote_average", "vote_count",
            "trailer_views", "trailer_likes", "trailer_comments",
            "trailer_popularity_index", "interaction_rate", "engagement_velocity",
            "youtube_sentiment", "sentiment_volatility",
            "trend_momentum",
            # Cast features
            "num_cast_members", "avg_cast_popularity", "max_cast_popularity", "star_power_score",
            # Director features
            "num_directors", "avg_director_popularity", "max_director_popularity", "director_experience_score",
            # Composer features
            "num_composers", "avg_composer_popularity", "max_composer_popularity", "music_prestige_score",
            # NEW: Engineered features
            "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
            "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
            # Text feature
            "overview",
            # Categorical features
            "primary_genre", "release_month"
        ]
        
        # Ensure features exist
        available_features = [f for f in features if f in df.columns]
        logger.info(f"Available features: {available_features}")
        
        X = df[available_features].copy()
        if 'overview' in X.columns:
            X['overview'] = X['overview'].fillna('')
            
        y = df["revenue"]
        
        return X, y

    def train_and_evaluate(self):
        X, y = self.load_data()
        if X is None:
            return

        # Preprocessing: OneHotEncode categorical features, TF-IDF for text
        # Identify categorical columns
        categorical_features = ['primary_genre', 'release_month']
        text_features = ['overview']
        # numeric is everything else
        numeric_features = [col for col in X.columns if col not in categorical_features and col not in text_features]
        
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('txt', TfidfVectorizer(max_features=100, stop_words='english'), 'overview')
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Create Pipeline
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            logger.info(f"{name} Results - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
            
            # Update best model tracking logic to save the PIPELINE not just the model
            if self.best_model_name == "" or results[name]["R2"] > results.get(self.best_model_name, {}).get("R2", -float('inf')):
                 self.best_model_name = name
                 self.best_model = clf

        logger.info(f"Best Regression Model: {self.best_model_name}")
        
        # Save Best Model Pipeline
        joblib.dump(self.best_model, os.path.join(config.MODELS_DIR, "best_regression_model.pkl"))

if __name__ == "__main__":
    regressor = RegressionModel()
    regressor.train_and_evaluate()
