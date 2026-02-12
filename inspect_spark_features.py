import pandas as pd
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def inspect():
    path = os.path.join(config.PROCESSED_DATA_DIR, "spark_features")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    try:
        df = pd.read_parquet(path)
        print("Columns in spark_features:")
        print(df.columns.tolist())
        
        if "primary_genre" in df.columns:
            print(f"primary_genre present. Sample: {df['primary_genre'].head().tolist()}")
        else:
            print("primary_genre MISSING")
            
        if "release_month" in df.columns:
            print(f"release_month present. Sample: {df['release_month'].head().tolist()}")
        else:
            print("release_month MISSING")
            
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    inspect()
