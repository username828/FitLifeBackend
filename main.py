# # backend/main.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Replace "*" with ["http://localhost:3000"] for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )




# # Define request body with the same fields as model_features
# class PredictionInput(BaseModel):
#     duration_minutes: float
#     activity_type: int
#     weight_kg: float
#     fitness_level: float
#     bmi: float
#     height_cm: float
#     intensity: int
#     gender: int

# @app.post("/predict")
# def predict(data: PredictionInput):
#     try:
#         # Maintain the exact feature order
#         input_array = np.array([[getattr(data, feature) for feature in model_features]])
#         scaled_input = scaler.transform(input_array)
#         prediction = model.predict(scaled_input)[0]
#         unscaled_prediction = prediction * original_std_calories + original_mean_calories

#         return {"predicted_calories_burned": unscaled_prediction}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn
# Your utility functions and objects (assumed to be defined somewhere)
# from your_module import predict_target, predict_health_condition, universal_scaler
# from your_module import original_std_calories, original_mean_calories, ...
# from your_module import label_encoder
# Load bundle (model, scaler, mean, std, features)
bundle = joblib.load("calories_burned_bundle.pkl")
#model = bundle["model"]
scaler = joblib.load("universal_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
#original_mean_calories = bundle["target_mean"]
# original_std_calories = bundle["target_std"]
original_mean_calories=15.381302193831326
original_std_calories=9.985552451615767


original_std_weight=22.46180081682238
original_mean_weight=94.92198077362113

original_std_fitness=5.502484562089024

original_mean_fitness=9.524899920168796

#model_features = bundle["features"]

# Pydantic models
class CaloriesInput(BaseModel):
    duration_minutes: float
    activity_type: int
    weight_kg: float
    fitness_level: float
    bmi: float
    height_cm: float
    intensity: int
    gender: int

class WeightInput(BaseModel):
    fitness_level: float
    bmi: float
    calories_burned: float
    gender: int
    daily_steps: float
    height_cm: float

class FitnessInput(BaseModel):
    weight_kg: float
    daily_steps: float
    calories_burned: float

class HealthInput(BaseModel):
    blood_pressure_systolic: float
    height_cm: float
    blood_pressure_diastolic: float
    resting_heart_rate: float
    bmi: float
    age: int

def predict_target(model, scaler, model_features, **kwargs):
    # Full feature list the scaler was trained on
    scaler_features = list(scaler.feature_names_in_)

    # Prepare input row for scaler (excluding unscaled features like 'gender')
    input_for_scaler = pd.DataFrame([np.zeros(len(scaler_features))], columns=scaler_features)

    for key, value in kwargs.items():
        if key in input_for_scaler.columns:
            input_for_scaler.at[0, key] = value

    # Scale
    scaled = scaler.transform(input_for_scaler)
    scaled_df = pd.DataFrame(scaled, columns=scaler_features)

    # Add non-scaled features (like 'gender') back
    for key, value in kwargs.items():
        if key not in scaler_features:
            scaled_df[key] = value

    # Final model input: only the features it was trained on
    model_input = scaled_df[model_features]

    return model.predict(model_input)[0]


def predict_health_condition(model, new_data, label_encoder):
    """
    Predict the health condition class for new input data using a trained XGBoost model.
    """
    # Ensure the input is in the right format
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data.values

    # Predict class index (e.g., 0, 1, 2, 3)
    predicted_class = model.predict(new_data)  # Returns array([class_index])
    predicted_probs = model.predict_proba(new_data)  # Array of probabilities
    condition_map = {
    0: "None",  # Key 0 for no condition
    1: "Hypertension",
    2: "Diabetes",
    3: "Asthma"  # Key 1 for any condition present
    }
    print(predicted_class, predicted_probs)
    return condition_map[predicted_class[0]], predicted_probs


# Initialize app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_calories = xgb.XGBRegressor()
model_calories.load_model("calories_burned_model.json")

model_weight = xgb.XGBRegressor()
model_weight.load_model("weight_model.json")

model_fitness = xgb.XGBRegressor()
model_fitness.load_model("fitness_model.json")

model_health = xgb.XGBClassifier()
model_health.load_model("health_condition_model.json")

# Prediction endpoints
@app.post("/predict/calories")
def predict_calories(data: CaloriesInput):
    model_features = [
        'duration_minutes', 'activity_type', 'weight_kg',
        'fitness_level', 'bmi', 'height_cm', 'intensity', 'gender'
    ]
    pred = predict_target(model_calories, scaler, model_features, **data.dict())
    unscaled = pred * original_std_calories + original_mean_calories
    
    return {"calories_burned": round(float(unscaled), 2)}


@app.post("/predict/weight")
def predict_weight(data: WeightInput):
    model_features = [
        'fitness_level', 'bmi', 'calories_burned',
        'gender', 'daily_steps', 'height_cm'
    ]
    pred = predict_target(model_weight, scaler, model_features, **data.dict())
    unscaled = pred * original_std_weight + original_mean_weight
    return {"weight_kg": round(float(unscaled), 2)}

@app.post("/predict/fitness")
def predict_fitness(data: FitnessInput):
    model_features = ['weight_kg', 'daily_steps', 'calories_burned']
    pred = predict_target(model_fitness, scaler, model_features, **data.dict())
    unscaled = pred * original_std_fitness + original_mean_fitness
    return {"fitness_level": round(float(unscaled), 4)}

@app.post("/predict/health-condition")
def predict_health(data: HealthInput):
    print(data.dict())  # Log the incoming data
    df = pd.DataFrame([data.dict()])
    label, probs = predict_health_condition(model_health, df, label_encoder)
    
    print(label)
    return {
        "predicted_condition": label,
        "probabilities": probs.tolist()  # Convert numpy array to list for JSON serialization
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)