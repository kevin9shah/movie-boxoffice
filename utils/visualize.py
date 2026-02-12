import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_visualizations():
    logger.info("Generating Visualizations...")
    
    dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv")
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found.")
        return

    df = pd.read_csv(dataset_path)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_DIR, "correlation_heatmap.png"))
    logger.info("Saved correlation_heatmap.png")

    # 2. Engagement vs Revenue (Scatter)
    if 'revenue' in df.columns and 'trailer_views' in df.columns:
        fig = px.scatter(df, x="trailer_views", y="revenue", 
                         title="Trailer Views vs Revenue",
                         hover_data=['title'] if 'title' in df.columns else None)
        fig.write_html(os.path.join(config.BASE_DIR, "engagement_revenue_scatter.html"))
        logger.info("Saved engagement_revenue_scatter.html")

    # 3. Sentiment Distribution
    if 'youtube_sentiment' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['youtube_sentiment'], kde=True, bins=20)
        plt.title("YouTube Sentiment Distribution")
        plt.savefig(os.path.join(config.BASE_DIR, "sentiment_dist.png"))
        logger.info("Saved sentiment_dist.png")

if __name__ == "__main__":
    generate_visualizations()
