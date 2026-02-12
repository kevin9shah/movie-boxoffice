from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from database.mongo_store import MongoStore

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Box Office Prediction API", version="1.0")

# Mount Static Files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Load Models
try:
    reg_model = joblib.load(os.path.join(config.MODELS_DIR, "best_regression_model.pkl"))
    cls_model = joblib.load(os.path.join(config.MODELS_DIR, "best_classification_model.pkl"))
    label_encoder = joblib.load(os.path.join(config.MODELS_DIR, "label_encoder.pkl"))
    mongo = MongoStore()
    logger.info("Models and DB loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or DB: {e}")
    reg_model = None
    cls_model = None
    label_encoder = None
    mongo = None

class MovieInput(BaseModel):
    title: str
    budget: float
    runtime: int
    popularity: float
    vote_average: float
    vote_count: int
    trailer_views: int
    trailer_likes: int
    trailer_comments: int
    trailer_popularity_index: float
    interaction_rate: float
    engagement_velocity: float
    youtube_sentiment: float
    sentiment_volatility: float
    trend_momentum: float
    primary_genre: str = "Action" # Default for simplicity in this demo
    release_month: int = 1

# --- Dashboard Endpoints ---

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/data/sample")
async def get_data_sample():
    """Return a sample of the processed data."""
    try:
        df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv"))
        df = df.fillna(0)
        # Return all data as requested previously, or a subset? User said "Spark Data does not show anything".
        # Let's return all but limited fields to keep it light? No, return all.
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/spark/stats")
async def get_spark_stats():
    """Return summary stats of the dataset."""
    try:
        df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv"))
        stats = {
            "total_movies": len(df),
            "avg_revenue": float(df['revenue'].mean()) if 'revenue' in df.columns else 0,
            "avg_budget": float(df['budget'].mean()) if 'budget' in df.columns else 0,
            "avg_sentiment": float(df['youtube_sentiment'].mean()) if 'youtube_sentiment' in df.columns else 0
        }
        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {"error": str(e)}

@app.get("/api/model/importance")
async def get_feature_importance():
    """Return feature importance from the regression model."""
    if not reg_model:
        return {"error": "Model not loaded"}
    
    try:
        # Access the classifier step from the pipeline
        if hasattr(reg_model, "named_steps") and "classifier" in reg_model.named_steps:
            model = reg_model.named_steps["classifier"]
            preprocessor = reg_model.named_steps["preprocessor"]
            
            # Get feature names from preprocessor
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                # Fallback if get_feature_names_out fails
                feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                
                # Clean up names
                clean_features = []
                for f in feature_names:
                    name = f.replace("num__", "").replace("cat__", "")
                    name = name.replace("primary_genre_", "Genre: ").replace("release_month_", "Month: ")
                    name = name.replace("_", " ").title()
                    clean_features.append(name)
                
                # Combine and sort
                feat_imp = [{"feature": f, "importance": float(i)} for f, i in zip(clean_features, importances)]
                return feat_imp
            elif hasattr(model, "coef_"):
                 return [{"feature": f, "importance": float(abs(i))} for f, i in zip(feature_names, model.coef_)]
            else:
                return []
        
        # Fallback for old non-pipeline models (unlikely now)
        elif hasattr(reg_model, "feature_importances_"):
             # ... (existing fallback logic if needed, or just return empty to be safe)
             return []
        else:
             return []
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/validation/report")
async def get_validation_report():
    """Return top 20 movies with validation results."""
    if not reg_model:
        return {"error": "Model not loaded"}

    try:
        df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_dataset.csv"))
        
        # Features (must match model)
        features = [
            "budget", "runtime", "popularity", "vote_average", "vote_count",
            "trailer_views", "trailer_likes", "trailer_comments",
            "trailer_popularity_index", "interaction_rate", "engagement_velocity",
            "youtube_sentiment", "sentiment_volatility",
            "trend_momentum",
            "primary_genre", "release_month"
        ]
        
        valid_df = df[df['revenue'] > 1000].copy()
        valid_df = valid_df.fillna(0)
        
        if valid_df.empty:
            return []
            
        X = valid_df[features]
        
        # Predict
        valid_df['predicted_revenue'] = reg_model.predict(X)
        valid_df['error_pct'] = ((valid_df['predicted_revenue'] - valid_df['revenue']).abs() / valid_df['revenue']) * 100
        
        # Sort by Revenue and take Top 20
        top_20 = valid_df.sort_values('revenue', ascending=False).head(20)
        
        return top_20[['title', 'revenue', 'predicted_revenue', 'error_pct']].to_dict(orient="records")

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}

@app.post("/predict")
def predict(movie: MovieInput):
    if not reg_model or not cls_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Prepare input dataframe
    # We need to handle the new features. 
    # For now, we will default them if not provided, basically assuming 'Action' and 'June' or similar?
    # Or strict validation. Let's strict.
    # Wait, MovieInput pydantic model needs update too.
    input_data = pd.DataFrame([movie.dict()])
    
    # Select features expected by model (excluding title)
    features = [
        "budget", "runtime", "popularity", "vote_average", "vote_count",
        "trailer_views", "trailer_likes", "trailer_comments",
        "trailer_popularity_index", "interaction_rate", "engagement_velocity",
        "youtube_sentiment", "sentiment_volatility",
        "trend_momentum",
        "primary_genre", "release_month"
    ]
    
    try:
        X = input_data[features]
        
        # Predict Revenue
        revenue_pred = reg_model.predict(X)[0]
        
        # Predict Success Class
        class_pred_idx = cls_model.predict(X)[0]
        success_class = label_encoder.inverse_transform([class_pred_idx])[0]
        
        # Confidence (Probability)
        try:
            proba = cls_model.predict_proba(X)[0]
            confidence = float(max(proba))
        except:
            confidence = 0.0

        # Save to DB
        if mongo:
            mongo.save_predictions(movie.title, revenue_pred, success_class, confidence)

        return {
            "title": movie.title,
            "predicted_revenue": float(revenue_pred),
            "success_class": success_class,
            "confidence_score": confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": reg_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
