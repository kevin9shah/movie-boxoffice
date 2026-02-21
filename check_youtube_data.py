import pandas as pd
import os
import sys
sys.path.append('.')
import config

df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'final_dataset.csv'))

print("="*70)
print("YOUTUBE & TRAILER DATA ANALYSIS")
print("="*70)

# Check for trailer data
print(f"\nTrailer data:")
print(f"  trailer_views: {(df['trailer_views'] > 0).sum()} movies with data")
print(f"  trailer_likes: {(df['trailer_likes'] > 0).sum()} movies with data")
print(f"  trailer_comments: {(df['trailer_comments'] > 0).sum()} movies with data")

# Check for youtube sentiment (likely from reviews)
print(f"\nYouTube sentiment data:")
print(f"  youtube_sentiment > 0: {(df['youtube_sentiment'] > 0).sum()} movies")
print(f"  Sentiment range: {df['youtube_sentiment'].min():.1f} - {df['youtube_sentiment'].max():.1f}")

# Check for interaction/engagement metrics
print(f"\nEngagement metrics:")
print(f"  interaction_rate > 0: {(df['interaction_rate'] > 0).sum()} movies")
print(f"  engagement_velocity > 0: {(df['engagement_velocity'] > 0).sum()} movies")
print(f"  trailer_popularity_index > 0: {(df['trailer_popularity_index'] > 0).sum()} movies")

# Cross-check: movies with zero trailer_views - do they have alternative YouTube data?
zero_trailer = df[df['trailer_views'] == 0]
print(f"\n{'='*70}")
print(f"Movies with ZERO trailer_views: {len(zero_trailer)}")
print(f"  - Have youtube_sentiment > 0: {(zero_trailer['youtube_sentiment'] > 0).sum()}")
print(f"  - Have engagement_velocity > 0: {(zero_trailer['engagement_velocity'] > 0).sum()}")
print(f"  - Have interaction_rate > 0: {(zero_trailer['interaction_rate'] > 0).sum()}")
print(f"  - Have trailer_popularity_index > 0: {(zero_trailer['trailer_popularity_index'] > 0).sum()}")

# Check what other metrics exist for zero-trailer movies
alt_signals = (
    (zero_trailer['youtube_sentiment'] > 0) | 
    (zero_trailer['engagement_velocity'] > 0) | 
    (zero_trailer['interaction_rate'] > 0)
)
print(f"  - Have ANY YouTube metric: {alt_signals.sum()}/{len(zero_trailer)}")

print("="*70)
