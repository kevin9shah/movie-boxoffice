import os
import sys
import logging
import argparse

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Import modules
from main_collection_pipeline import main as run_collection
from bigdata_processing.spark_cleaning import main as run_spark_cleaning
from nlp.sentiment_analysis import SentimentAnalyzer
from bigdata_processing.spark_feature_engineering import run_feature_engineering
from feature_engineering.integrate_features import integrate_features
from models.regression_models import RegressionModel
from models.classification_models import ClassificationModel
from models.model_evaluation import evaluate_models
from utils.visualize import generate_visualizations

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(skip_collection=False):
    logger.info("==================================================")
    logger.info("   Starting Movie Box Office Prediction System    ")
    logger.info("==================================================")

    # 1. Data Collection
    if not skip_collection:
        logger.info("\n[STEP 1] Data Collection (TMDB, YouTube, Trends, Wikipedia)")
        try:
            run_collection()
        except Exception as e:
            logger.error(f"Data Collection Failed: {e}")
            return
    else:
        logger.info("\n[STEP 1] Data Collection (Skipped)")

    # 2. Big Data Processing (Cleaning)
    logger.info("\n[STEP 2] Big Data Processing (Cleaning)")
    try:
        run_spark_cleaning()
    except Exception as e:
        logger.error(f"Spark Cleaning Failed: {e}")
        return

    # 3. NLP Analysis
    logger.info("\n[STEP 3] NLP Analysis")
    try:
        analyzer = SentimentAnalyzer()
        analyzer.run()
    except Exception as e:
        logger.error(f"NLP Analysis Failed: {e}")
        return

    # 4. Big Data Processing (Feature Engineering)
    logger.info("\n[STEP 4] Big Data Feature Engineering")
    try:
        run_feature_engineering()
    except Exception as e:
        logger.error(f"Spark Feature Engineering Failed: {e}")
        return
        
    # 5. Feature Integration
    logger.info("\n[STEP 5] Feature Integration")
    try:
        integrate_features()
    except Exception as e:
        logger.error(f"Feature Integration Failed: {e}")
        return

    # 6. Model Training & Evaluation
    logger.info("\n[STEP 6] Model Training")
    try:
        logger.info("Training Regression Models...")
        reg = RegressionModel()
        reg.train_and_evaluate()
        
        logger.info("Training Classification Models...")
        cls = ClassificationModel()
        cls.train_and_evaluate()
        
        evaluate_models()
    except Exception as e:
        logger.error(f"Model Training Failed: {e}")
        return

    # 7. Visualization
    logger.info("\n[STEP 7] generating Visualizations")
    try:
        generate_visualizations()
    except Exception as e:
        logger.error(f"Visualization Failed: {e}")
        return

    logger.info("\n==================================================")
    logger.info("           Pipeline Completed Successfully!       ")
    logger.info("==================================================")
    logger.info("To start the API, run: uvicorn api.app:app --reload")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection step")
    args = parser.parse_args()
    
    run_pipeline(skip_collection=args.skip_collection)
