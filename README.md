# Movie Box Office Success Prediction System

## Overview
This is a production-level Big Data Analytics system designed to predict:
- Opening weekend revenue
- Total box office revenue
- Movie success classification (Hit / Average / Flop)

The system leverages multi-source big data including:
- **TMDb**: Movie metadata (budget, genre, cast, etc.)
- **YouTube**: Trailer statistics and engagement.
- **Reddit**: Community sentiment and buzz.
- **Google Trends**: Search interest momentum.

## Architecture
1. **Data Collection**: Python scripts to fetch data from APIs and save as raw JSON.
2. **Big Data Processing**: PySpark for cleaning, normalization, and aggregation.
3. **NLP**: Sentiment analysis on comments using VADER and Transformers.
4. **Feature Engineering**: Integration of all data sources into a final dataset.
5. **Machine Learning**: Regression and Classification models (Linear, RF, XGBoost).
6. **API**: FastAPI endpoint for serving predictions.
7. **Dashboard**: Visualization of insights and model performance.

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Create a `.env` file in the root directory with the following keys:
   ```
   TMDB_API_KEY=your_key
   YOUTUBE_API_KEY=your_key
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   MONGO_URI=mongodb://localhost:27017/
   ```
3. **Run Pipeline**:
   ```bash
   python main_pipeline.py
   ```

## Folder Structure
- `data/`: Raw and processed data storage.
- `data_collection/`: Scripts for fetching data.
- `bigdata_processing/`: PySpark jobs.
- `nlp/`: Sentiment analysis modules.
- `models/`: ML model training and evaluation.
- `api/`: FastAPI application.
# movie-boxoffice
