from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.Catboost_bjørk import predict_pollen as predict_bjørk
from models.Catboost_gress import predict_pollen as predict_gress

app = FastAPI()


class PollenRequest(BaseModel):
    pollen_type: str
    today_date: str


@app.post("/predict")
def predict(request: PollenRequest):
    pollen_type = request.pollen_type.strip().casefold()

    if pollen_type == "bjørk":
        forecast = predict_bjørk(today_date=request.today_date)
    elif pollen_type == "gress":
        forecast = predict_gress(today_date=request.today_date)
    else:
        raise HTTPException(
            status_code=400,
            detail="Ugyldig pollen_type. Bruk 'bjørk' eller 'gress'.",
        )

    return {"forecast": forecast}