import pandas as pd
import os
import sys
import numpy as np
sys.path.append('.')
import config

df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'final_dataset.csv'))
print(f"Before: {len(df)} movies")
print(f"Movies with trailers (trailer_views > 0): {(df['trailer_views'] > 0).sum()}")

# Strategy: Keep movies with trailers OR very high popularity
# High popularity suggests mainstream movies with good prediction signals
df_clean = df[
    (df['trailer_views'] > 0) |  # Movies with trailer data
    (df['popularity'] >= 40)      # OR very popular movies (strong signals)
].copy()

print(f"After filter (trailers OR popularity>=40): {len(df_clean)} movies")
print(f"Removed: {len(df) - len(df_clean)} movies")

# For remaining movies with zero trailers but high popularity, impute trailers
# based on their popularity and vote average
zero_trailer_mask = (df_clean['trailer_views'] == 0)
if zero_trailer_mask.sum() > 0:
    print(f"\nImputing trailer data for {zero_trailer_mask.sum()} movies...")
    # Estimate: trailer_views ~= popularity * vote_average * 5000
    df_clean.loc[zero_trailer_mask, 'trailer_views'] = (
        df_clean.loc[zero_trailer_mask, 'popularity'] * 
        df_clean.loc[zero_trailer_mask, 'vote_average'] * 
        5000
    )
    print(f"Imputed trailer_views: min={df_clean['trailer_views'].min():.0f}, max={df_clean['trailer_views'].max():.0f}")

# Save cleaned dataset
output_path = os.path.join(config.PROCESSED_DATA_DIR, 'final_dataset.csv')
df_clean.to_csv(output_path, index=False)
print(f"\nDataset saved: {len(df_clean)} movies")
print(f"Movies with zero trailers (after imputation): {(df_clean['trailer_views'] == 0).sum()}")
