from fastapi import FastAPI
from pydantic import BaseModel

# Denne funksjonen må finnes i catboostregression.py.
from models.Catboost_bjørk import predict_pollen

app = FastAPI()


class PollenRequest(BaseModel):
    pollen_type: str
    today_date: str
    location: str


@app.post("/predict")
def predict(request: PollenRequest):
    forecast = predict_pollen(
        pollen_type=request.pollen_type,
        today_date=request.today_date,
        location=request.location,
    )

    return {"forecast": forecast}

