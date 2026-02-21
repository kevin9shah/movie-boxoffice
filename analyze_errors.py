import json, statistics

with open('final_report.json', 'r') as f:
    data = json.load(f)

all_errors = [m['error_percentage'] for m in data['results']]
top_100 = data['results'][:100]
top_100_errors = [m['error_percentage'] for m in top_100]
bottom = data['results'][100:]
bottom_errors = [m['error_percentage'] for m in bottom]

print("Top 100 movies mean error:", statistics.mean(top_100_errors))
print("Bottom", len(bottom), "movies mean error:", statistics.mean(bottom_errors))
print("Total all movies mean error:", statistics.mean(all_errors))
