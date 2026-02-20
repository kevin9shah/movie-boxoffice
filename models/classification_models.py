import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
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

class ClassificationModel:
    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""
        self.le = LabelEncoder()

    def load_data(self):
        dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
        if not os.path.exists(dataset_path):
            logger.error("Dataset not found.")
            return None, None
            
        df = pd.read_csv(dataset_path)
        
        # Select Features (same as regression)
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
            # Engineered features
            "is_franchise", "is_sequel", "budget_tier", "genre_avg_revenue",
            "description_length", "hype_score", "budget_popularity_ratio", "vote_power",
            # Text feature
            "overview",
            # Categorical features
            "primary_genre", "release_month"
        ]
        
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].copy()
        
        if 'overview' in X.columns:
            X['overview'] = X['overview'].fillna('')
        
        # Target
        if "success_class" not in df.columns:
            logger.error("Target column 'success_class' not found.")
            return None, None
            
        y = self.le.fit_transform(df["success_class"])
        
        return X, y

    def train_and_evaluate(self):
        X, y = self.load_data()
        if X is None:
            return

        # Preprocessing
        categorical_features = ['primary_genre', 'release_month']
        text_features = ['overview']
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
            
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            try:
                y_prob = clf.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except:
                auc = 0.0
            
            results[name] = {"Accuracy": acc, "F1": f1, "AUC": auc}
            logger.info(f"{name} Results - Accuracy: {acc}, F1: {f1}, AUC: {auc}")

            if self.best_model_name == "" or results[name]["F1"] > results.get(self.best_model_name, {}).get("F1", -1):
                 self.best_model_name = name
                 self.best_model = clf

        logger.info(f"Best Classification Model: {self.best_model_name}")
        
        # Save Best Model and Encoder
        joblib.dump(self.best_model, os.path.join(config.MODELS_DIR, "best_classification_model.pkl"))
        joblib.dump(self.le, os.path.join(config.MODELS_DIR, "label_encoder.pkl"))

if __name__ == "__main__":
    classifier = ClassificationModel()
    classifier.train_and_evaluate()
