import pymongo
import logging
import sys
import os
import json

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MongoStore:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(config.MONGO_URI)
            self.db = self.client[config.DB_NAME]
            logger.info(f"Connected to MongoDB: {config.DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None

    def insert_raw_data(self, collection_name, data):
        """Insert raw data into a specific collection."""
        if not self.db:
            return
            
        collection = self.db[collection_name]
        try:
            if isinstance(data, list):
                if data:
                    collection.insert_many(data)
            else:
                collection.insert_one(data)
            logger.info(f"Inserted data into {collection_name}")
        except Exception as e:
            logger.error(f"Error inserting into {collection_name}: {e}")

    def save_predictions(self, title, revenue_pred, success_class, confidence=None):
        """Save prediction results."""
        if not self.db:
            return
            
        collection = self.db["predictions"]
        record = {
            "title": title,
            "predicted_revenue": revenue_pred,
            "success_class": success_class,
            "confidence": confidence
        }
        try:
            collection.insert_one(record)
            logger.info(f"Saved prediction for {title}")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

if __name__ == "__main__":
    # Test connection
    store = MongoStore()
