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
            "budget", "runtime", "popularity", "vote_average", "vote_count",
            "trailer_views", "trailer_likes", "trailer_comments",
            "trailer_popularity_index", "interaction_rate", "engagement_velocity",
            "youtube_sentiment", "sentiment_volatility",
            "trend_momentum",
            "primary_genre", "release_month"
        ]
        
        available_features = [f for f in features if f in df.columns]
        X = df[available_features]
        
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
