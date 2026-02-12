import joblib
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_feature_importance(model_path, output_name):
    if not os.path.exists(model_path):
        return

    model = joblib.load(model_path)
    
    # Try to get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # We need feature names. Since we don't save them in the model object easily without a pipeline, 
        # we'll reload the dataset columns to map them.
        dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
        df = pd.read_csv(dataset_path)
        features = [
            "budget", "runtime", "popularity", "vote_average", "vote_count",
            "trailer_views", "trailer_likes", "trailer_comments",
            "trailer_popularity_index", "interaction_rate", "engagement_velocity",
            "youtube_sentiment", "sentiment_volatility",
            "trend_momentum"
        ]
        available_features = [f for f in features if f in df.columns]
        
        if len(importances) != len(available_features):
            logger.warning("Feature count mismatch. Skipping plot.")
            return

        feature_imp = pd.Series(importances, index=available_features).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.title(f"Feature Importance - {output_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_DIR, f"{output_name}_feature_importance.png"))
        logger.info(f"Saved feature importance plot to {output_name}_feature_importance.png")
    else:
        logger.info(f"Model {output_name} does not support feature_importances_.")

def evaluate_models():
    logger.info("Evaluating Models and Generating Plots...")
    
    reg_model_path = os.path.join(config.MODELS_DIR, "best_regression_model.pkl")
    cls_model_path = os.path.join(config.MODELS_DIR, "best_classification_model.pkl")
    
    plot_feature_importance(reg_model_path, "Regression")
    plot_feature_importance(cls_model_path, "Classification")

if __name__ == "__main__":
    evaluate_models()
