import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List

# Define a Pydantic model for the request body
# This class will define the expected data format for our API
class DataInput(BaseModel):
    PAY_0: float
    AGE: float
    LIMIT_BAL: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    PAY_AMT1: float
    BILL_AMT6: float
    BILL_AMT4: float
    BILL_AMT5: float
    PAY_2: float
    PAY_AMT6: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float

# Load the trained model
model = joblib.load('rf_final.joblib')

# Initialize the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: DataInput):
    try:
        # Convert the incoming data to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Make predictions
        predictions = model.predict(input_df)

        # Return the prediction as a JSON response
        response = {"prediction": predictions.tolist()[0]}
        return response

    except Exception as e:
        return {"error": str(e)}