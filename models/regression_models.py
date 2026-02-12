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
        
        # Select Features
        features = [
            "budget", "runtime", "popularity", "vote_average", "vote_count",
            "trailer_views", "trailer_likes", "trailer_comments",
            "trailer_popularity_index", "interaction_rate", "engagement_velocity",
            "youtube_sentiment", "sentiment_volatility",
            "trend_momentum",
            "primary_genre", "release_month"
        ]
        
        # Ensure features exist
        available_features = [f for f in features if f in df.columns]
        
        X = df[available_features]
        y = df["revenue"]
        
        return X, y

    def train_and_evaluate(self):
        X, y = self.load_data()
        if X is None:
            return

        # Preprocessing: OneHotEncode categorical features
        # Identify categorical columns
        categorical_features = ['primary_genre', 'release_month']
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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
