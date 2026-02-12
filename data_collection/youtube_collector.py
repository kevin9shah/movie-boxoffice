import os
import json
import logging
import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class YouTubeCollector:
    def __init__(self):
        self.api_key = config.YOUTUBE_API_KEY
        if self.api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        else:
            logger.warning("YouTube API Key not found in config.")
            self.youtube = None

    def search_trailer(self, movie_title):
        """Search for the official trailer of a movie."""
        if not self.youtube:
            return []
            
        query = f"{movie_title} official trailer"
        try:
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=5, # Fetch top 5 candidates
                type='video'
            ).execute()
            
            return search_response.get('items', [])
        except HttpError as e:
            logger.error(f"Error searching for trailer {movie_title}: {e}")
            return []

    def get_video_stats(self, video_id):
        """Fetch statistics for a specific video."""
        if not self.youtube:
            return None
            
        try:
            video_response = self.youtube.videos().list(
                id=video_id,
                part='statistics,contentDetails'
            ).execute()
            
            if video_response.get('items'):
                return video_response['items'][0]['statistics']
            return None
        except HttpError as e:
            logger.error(f"Error getting stats for video ID {video_id}: {e}")
            return None
            
    def get_comments(self, video_id, max_results=50):
        """Fetch top comments for a video."""
        if not self.youtube:
            return []
            
        try:
            comment_response = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText",
                order="relevance"
            ).execute()
            
            comments = []
            for item in comment_response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            return comments
        except HttpError as e:
            logger.warning(f"Could not fetch comments for video ID {video_id}: {e}")
            return []

    def save_data(self, data, filename):
        """Save data to JSON file."""
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def run(self, movie_list):
        """Main execution method taking a list of movie dictionaries (from TMDB)."""
        logger.info("Starting YouTube Data Collection...")
        youtube_data = []
        
        for movie in movie_list:
            title = movie.get('title')
            movie_id = movie.get('id')
            
            logger.info(f"Processing trailer for: {title}")
            candidates = self.search_trailer(title)
            
            selected_video_id = None
            selected_stats = None
            
            # Iterate through candidates to find one with comments
            for video in candidates:
                vid = video['id']['videoId']
                stats = self.get_video_stats(vid)
                
                if stats:
                    comment_count = int(stats.get('commentCount', 0))
                    if comment_count > 0:
                        selected_video_id = vid
                        selected_stats = stats
                        logger.info(f"Selected video {vid} with {comment_count} comments.")
                        break
            
            # Fallback to first video if no comments found (or if all disabled)
            if not selected_video_id and candidates:
                logger.warning(f"No video with comments found for {title}. Falling back to top result.")
                selected_video_id = candidates[0]['id']['videoId']
                selected_stats = self.get_video_stats(selected_video_id)

            if selected_video_id and selected_stats:
                comments = self.get_comments(selected_video_id)
                
                entry = {
                    "movie_id": movie_id,
                    "title": title,
                    "video_id": selected_video_id,
                    "trailer_stats": selected_stats,
                    "comments": comments
                }
                youtube_data.append(entry)
            else:
                logger.warning(f"No valid trailer found for {title}")
        
        self.save_data(youtube_data, "youtube_trailers.json")
        logger.info(f"Completed! Collected data for {len(youtube_data)} trailers.")

if __name__ == "__main__":
    # Dummy data for testing standalone run
    dummy_movies = [{"id": 550, "title": "Fight Club"}, {"id": 27205, "title": "Inception"}]
    collector = YouTubeCollector()
    collector.run(dummy_movies)
