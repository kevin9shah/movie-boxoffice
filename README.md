# Big Data Driven Multi-Modal Movie Box Office Prediction System

**Document No:** 02-IPR-R003  
**Issue No/Date:** 2/01.02.2024  
**Amd. No/Date:** 0/00.00.0000

---

## 1. Title of the invention
**Big Data Driven Multi-Modal Movie Box Office Prediction System**

## 2. Field / Area of invention
**Field**: Computer Science & Engineering / Data Science  
**Area**: Big Data Analytics, Machine Learning, Predictive Modeling, Natural Language Processing (Sentiment Analysis).

## 3. Prior Patents and Publications from literature

| S. No | Patent / Publication Title | Year | Summary of Technique | Limitation Addressed by Proposed System |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Traditional Statistical Box Office Forecasting** | 2010 | Uses historical averages and basic regression on budget/screens. | Ignores the "hype" factor and social media sentiment which are crucial drivers today. |
| 2 | **Social Media Analytics for Movie Success** | 2015 | Analyzes Twitter/Facebook likes to predict opening weekends. | Often relies on small datasets or single platforms; lacks robust Big Data processing for scalability. |
| 3 | **Cast & Crew Based Prediction Models** | 2018 | Assigns "star power" scores to actors/directors to predict revenue. | Fails to account for "sleeper hits" or "viral flops" driven by audience reception (trailers, reviews). |
| 4 | **Proposed System** | **2026** | **Hybrid Big Data Pipeline (PySpark + ML + NLP + TMDB Credits)** | Integrates **Structured** (Meta-data, Cast/Crew Credits), **Unstructured** (Comments/Reviews), and **Derived** (Engineered Features) data using a scalable Spark architecture. |

## 4. Summary and background of the invention (Address the gap / Novelty)
The motion picture industry is a high-risk, high-reward sector where accurate revenue prediction can save millions in marketing and distribution costs. 

**The Gap**: Existing solutions typically focus on *either* financial metrics (budget) *or* social metrics (likes), rarely integrating them effectively at scale. They also struggle with the volume of unstructured data (thousands of comments) and data quality issues like duplicates.

**The Novelty**: This invention proposes a **Unified Big Data Pipeline** that:
1.  Ingests data from multiple heterogeneous sources (TMDB API, YouTube API, Google Search).
2.  Uses **Apache Spark (PySpark)** for parallel processing and robust deduplication of large-scale movie data.
3.  **Enriches the dataset with real cast/crew credits** from TMDB, computing Star Power Score, Director Popularity, and Music Prestige features for all 674 movies.
4.  Engineers novel features like `Sentiment Volatility`, `Engagement Velocity`, `Hype Score`, `Vote Power`, and `Budget Popularity Ratio`.
5.  Combines Regression (Revenue) and Classification (Hit/Flop) models for a holistic predictive view.
6.  **Eliminates data leakage** by removing Wikipedia features that were correlated with the target variable.

## 5. Objective(s) of Invention
1.  To develop a scalable Big Data architecture for ingesting and processing movie metadata and social feedback.
2.  To improve prediction accuracy (minimize MAPE) for high-grossing films by incorporating "hype" metrics and star power features.
3.  To provide a real-time, clean, and professional analytics dashboard with model evaluation scores for stakeholders.
4.  To demonstrate the use of PySpark for cleaning (deduplication) and feature engineering on disparate datasets.

## 6. Working principle of the invention (in brief)
The system operates on a **Data Lakehouse** principle:
1.  **Ingestion**: Collectors fetch raw data from APIs (TMDB, YouTube, Google Trends).
2.  **Enrichment**: TMDB Credits API enriches each movie with cast/crew popularity, star power scores, and director experience data.
3.  **Processing**: A Spark Cluster cleans, **deduplicates**, and joins the data. NLP algorithms (TextBlob/VADER) quantify public sentiment.
4.  **Feature Engineering**: 8 novel features are derived: `is_franchise`, `is_sequel`, `budget_tier`, `genre_avg_revenue`, `description_length`, `hype_score`, `budget_popularity_ratio`, `vote_power`.
5.  **Learning**: The processed features (37 total) are fed into Ensemble ML models (LightGBM/Random Forest/XGBoost) with sample weighting to prioritize blockbuster accuracy.
6.  **Serving**: A FastAPI backend serves predictions and model evaluation scores to a web dashboard.

## 7. Description of the invention in detail (Workflow A to Z)

### A. System Architecture
The system is composed of four main layers:
1.  **Data Collection Layer**: Python scripts (`data_collection/`) fetching live data from TMDB, YouTube, and Google Search APIs.
2.  **Data Enrichment Layer**: `enrich_and_retrain.py` fetches TMDB credits for all movies, computing cast/crew features with caching.
3.  **Big Data Processing Layer**: PySpark jobs (`bigdata_processing/`) for ETL (Extract, Transform, Load).
4.  **Machine Learning Layer**: Scikit-Learn pipelines (`models/`) for training, evaluation, and inference using LightGBM.
5.  **Application Layer**: FastAPI + HTML/JS (`api/`) for the user interface and model evaluation dashboard.

### B. Detailed Workflow (A to Z)

**Step 1: Data Acquisition**
-   **TMDB API**: Fetches core metadata (Budget, Runtime, Release Date, Genres) and movie credits (Cast, Directors, Composers).
-   **YouTube API**: Searches for official trailers, fetching views, likes, comments, and engagement metrics.
-   **Google Search (Optional)**: Scrapes review snippets for external validation.
-   *Outcome*: Raw JSON files stored in `data/raw/`.

**Step 2: Data Enrichment**
-   **TMDB Credits API**: For each of 674 movies, fetches top 10 cast members, directors, and composers with their popularity scores.
-   **Derived Features**: Computes `star_power_score` (sum of top 3 cast popularity), `director_experience_score`, `music_prestige_score`.
-   **Caching**: Credits are cached in `data/raw/credits_cache.json` to avoid redundant API calls.

**Step 3: Spark Processing (Big Data)**
-   **Cleaning**: `spark_cleaning.py` loads raw data into Spark DataFrames, handles missing values, and casts types.
-   **Deduplication**: Removes duplicate movie entries based on ID.
-   **Feature Engineering**: `spark_feature_engineering.py` derives:
    -   `release_month` (Seasonality), `primary_genre` (Categorical encoding)
    -   `engagement_ratio` (Likes / Views), `interaction_rate`, `engagement_velocity`
    -   8 engineered features: `is_franchise`, `is_sequel`, `budget_tier`, `genre_avg_revenue`, `description_length`, `hype_score`, `budget_popularity_ratio`, `vote_power`

**Step 4: Sentiment Analysis**
-   TextBlob is applied to YouTube comments to calculate `polarity` (-1 to +1) and `subjectivity`.
-   Scores are aggregated to create `youtube_sentiment`, `sentiment_volatility`, and `trend_momentum` indices.

**Step 5: Model Training**
-   **Preprocessing**: `ColumnTransformer` applies OneHotEncoding to Genre/Month, StandardScaler to numerics, TF-IDF (200 features) to movie overviews.
-   **Regression**: LightGBM Regressor predicts exact Box Office Revenue ($) — trained with sample weights to prioritize blockbusters.
-   **Classification**: Random Forest Classifier predicts Success Class (Flop, Moderate, Hit, Blockbuster).
-   **Data Leakage Prevention**: Wikipedia features (`wikipedia_worldwide_box_office`, `wikipedia_budget`) are excluded from model training as they correlate directly with revenue.
-   **Validation**: Models are evaluated using R², MAE, MAPE, and RMSE.

**Step 6: Deployment & Visualization**
-   **API**: REST endpoints serve data (`/api/data/sample`), predictions (`/predict`), model scores (`/api/model/scores`), and validation (`/api/validation/report`).
-   **Dashboard**: A professional, clean UI (Inter font, light theme) displays:
    -   **Model Performance Scores**: R², MAE, MAPE (Overall & Top 50), RMSE, Model Type, Feature Groups.
    -   Key Performance Indicators (Avg Revenue, Sentiment).
    -   Feature Importance Charts (Plotly.js).
    -   Real-time Validation Reports comparing Actual vs Predicted revenue (Top 100 of 674 movies).
    -   **Prediction Simulator** with Cast/Crew stats, Sentiment slider, and NLP analysis.

### C. Flowchart / Diagram
*(Conceptual)*
`Raw Data (JSON)` -> `TMDB Credits Enrichment` -> `PySpark ETL` -> `Feature Engineering (37 features)` -> `LightGBM Train (with Sample Weights)` -> `Model (.pkl)` -> `FastAPI` -> `User Dashboard (with Model Scores)`

## 8. How to run

### Prerequisites
- Python 3.10+
- TMDB API Key (set in `.env` as `TMDB_API_KEY`)

### Setup
```bash
cd movie_box_office_prediction
python -m venv venv
source venv/bin/activate      # On Mac/Linux
pip install -r requirements.txt
```

### Step 1: Data Collection
```bash
python data_collection/tmdb_collector.py
python data_collection/youtube_collector.py
```

### Step 2: Data Enrichment (Cast/Crew from TMDB Credits)
```bash
python enrich_and_retrain.py
```
This script:
- Fetches cast/crew credits for all movies in the dataset via TMDB API
- Caches results in `data/raw/credits_cache.json`
- Computes star power, director popularity, and composer scores
- Removes data leakage features (wikipedia box office)
- Retrains the model with sample weights (LightGBM/XGBoost/RandomForest)
- Prints validation results for the top 20 movies by revenue

### Step 3: Big Data Processing (PySpark)
```bash
python bigdata_processing/spark_cleaning.py
python bigdata_processing/spark_feature_engineering.py
```

### Step 4: Model Training (if not done via enrich_and_retrain.py)
```bash
python models/regression_models.py
python models/classification_models.py
```

### Step 5: Start the Dashboard
```bash
uvicorn api.app:app --reload
```
Open http://localhost:8000/dashboard in your browser.

## 9. Experimental validation results

The system was validated against a dataset of **674 movies** (post-deduplication) with **37 features** including enriched cast/crew data from TMDB.

**Model**: LightGBM with sample weights (blockbuster-prioritized training)

**Key Results:**
-   **R² Score**: 0.53+ (on full dataset without data leakage)
-   **Model features**: 37 (Financial, Audience, Trailer, Sentiment, Cast/Crew, Engineered)
-   **Data leakage eliminated**: Removed `wikipedia_worldwide_box_office` (was identical to revenue)
-   **All scores visible on dashboard** at `/dashboard` → Model Scores section

## 10. What aspect(s) of the invention need(s) protection?
1.  **The Multi-Modal Feature Integration Logic**: The weighted combination of `Trailer Engagement Velocity`, `Sentiment Volatility`, and `Star Power Score` with traditional `Budget` metrics.
2.  **The Spark-Based Sentiment Pipeline**: The methodology of aggregating localized comment sentiment into a global movie score at scale.
3.  **The Cast/Crew Enrichment Pipeline**: Real-time TMDB Credits enrichment with caching and star power computation.

## 11. Technology readiness level (TRL)

**Current Status**: **TRL 4 (Technology validated in a lab)**

-   [ ] TRL 1: Basic Principles observed
-   [ ] TRL 2: Technology concept formulated
-   [ ] TRL 3: Experimental proof of concept
-   [x] **TRL 4: Technology validated in a lab** (Working prototype with real-world data)
-   [ ] TRL 5: Technology validated in a relevant environment
-   [ ] TRL 6: Technology demonstrated in a relevant environment
-   [ ] TRL 7: System prototype demonstration in an operational environment
-   [ ] TRL 8: System complete and qualified
-   [ ] TRL 9: Actual system proven in an operational environment

---
*End of Document*
