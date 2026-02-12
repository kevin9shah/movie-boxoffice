import pandas as pd
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def remove_duplicates():
    input_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    if not os.path.exists(input_path):
        print("Dataset not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Remove duplicates based on movie_id
    df_clean = df.drop_duplicates(subset=['movie_id'], keep='first')
    print(f"Cleaned shape: {df_clean.shape}")
    
    df_clean.to_csv(input_path, index=False)
    print(f"Saved cleaned dataset to {input_path}")

if __name__ == "__main__":
    remove_duplicates()
