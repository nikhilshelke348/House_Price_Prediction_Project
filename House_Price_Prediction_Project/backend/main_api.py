from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = FastAPI(
    title="ValuHome AI | Real Estate Valuation Backend API",
    description="REST API backend for Machine Learning House Price Prediction in Indian Rupees (₹)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(base_dir, '..', 'models'))
MODEL_FILE = os.path.join(models_dir, 'house_price_model.pkl')

if not os.path.exists(MODEL_FILE):
    sys.path.append(models_dir)
    from train_model import train_and_evaluate
    train_and_evaluate()

model_bundle = joblib.load(MODEL_FILE)
pipeline = model_bundle['model_pipeline']
best_model_name = model_bundle['best_model_name']
metrics = model_bundle['metrics']
feature_importance = model_bundle.get('feature_importance', [])
rmse = metrics[best_model_name]['RMSE']

class HouseFeatures(BaseModel):
    square_feet: int = Field(..., example=2000, description="Built-up living area in sq. ft.")
    bedrooms: int = Field(..., example=3, description="Number of bedrooms (BHK)")
    bathrooms: float = Field(..., example=2.0, description="Number of bathrooms")
    age_years: int = Field(..., example=8, description="Age of house in years")
    garage_cars: int = Field(..., example=2, description="Parking capacity for cars")
    location: str = Field(..., example="Suburbs", description="Location: Suburbs, Downtown, Rural, Prime Coastal")
    overall_condition: int = Field(..., example=7, description="Condition rating on 1-10 scale")

def format_inr(val: float) -> str:
    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} Lakhs"
    else:
        return f"₹{val:,.0f}"

@app.get("/")
def read_root():
    return {
        "service": "ValuHome AI House Price Prediction Backend",
        "status": "online",
        "deployed_model": best_model_name,
        "r2_accuracy": metrics[best_model_name]['R2'],
        "docs_url": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    return {
        "deployed_model": best_model_name,
        "model_benchmarks": metrics
    }

@app.get("/features")
def get_features():
    return {
        "deployed_model": best_model_name,
        "feature_importance": feature_importance
    }

@app.post("/predict")
def predict_price(features: HouseFeatures):
    allowed_locations = ['Suburbs', 'Downtown', 'Rural', 'Prime Coastal']
    if features.location not in allowed_locations:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid location '{features.location}'. Must be one of {allowed_locations}"
        )
    
    input_df = pd.DataFrame([{
        'Square_Feet': features.square_feet,
        'Bedrooms': features.bedrooms,
        'Bathrooms': features.bathrooms,
        'Age_Years': features.age_years,
        'Garage_Cars': features.garage_cars,
        'Location': features.location,
        'Overall_Condition': features.overall_condition
    }])
    
    try:
        predicted_price = float(pipeline.predict(input_df)[0])
        min_range = max(0, predicted_price - rmse)
        max_range = predicted_price + rmse
        
        return {
            "success": True,
            "prediction": {
                "estimated_price_raw": round(predicted_price, 2),
                "estimated_price_formatted": format_inr(predicted_price),
                "price_per_sqft": round(predicted_price / features.square_feet, 2),
                "confidence_interval": {
                    "min_price": round(min_range, 2),
                    "min_price_formatted": format_inr(min_range),
                    "max_price": round(max_range, 2),
                    "max_price_formatted": format_inr(max_range),
                    "rmse_margin": round(rmse, 2)
                }
            },
            "model_info": {
                "name": best_model_name,
                "r2_score": metrics[best_model_name]['R2']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)
