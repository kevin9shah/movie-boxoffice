import json
import random
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def simulate_data():
    tmdb_path = os.path.join(config.RAW_DATA_DIR, "tmdb_movies.json")
    if not os.path.exists(tmdb_path):
        print("TMDB data not found.")
        return

    with open(tmdb_path, 'r') as f:
        movies = json.load(f)

    youtube_data = []
    print(f"Simulating YouTube data for {len(movies)} movies...")

    for movie in movies:
        # Simulate stats based on popularity to make it somewhat realistic
        pop = movie.get('popularity', 10)
        views = int(pop * random.uniform(5000, 50000))
        likes = int(views * random.uniform(0.02, 0.05))
        comments = int(views * random.uniform(0.001, 0.01))
        
        entry = {
            "movie_id": movie.get('id'),
            "title": movie.get('title'),
            "cat_video_id": "simulated_id",
            "trailer_stats": {
                "viewCount": str(views),
                "likeCount": str(likes),
                "commentCount": str(comments)
            },
            "comments": ["This movie looks great!", "Can't wait to see it.", "Looks okay I guess."]
        }
        youtube_data.append(entry)

    output_path = os.path.join(config.RAW_DATA_DIR, "youtube_trailers.json")
    with open(output_path, 'w') as f:
        json.dump(youtube_data, f, indent=4)
    
    print(f"Saved simulated data to {output_path}")

if __name__ == "__main__":
    simulate_data()
