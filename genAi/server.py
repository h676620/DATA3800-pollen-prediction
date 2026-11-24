from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.pollen_lookup import lookup_pollen

app = FastAPI()


class PollenRequest(BaseModel):
    pollen_type: str
    today_date: str


@app.post("/predict")
def predict(request: PollenRequest):
    try:
        datetime.strptime(request.today_date, "%d.%m.%Y")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Dato må ha formatet DD.MM.YYYY.",
        )

    pollen_type = request.pollen_type.strip().casefold()

    try:
        result = lookup_pollen(pollen_type, request.today_date)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {
        "pollen_type": pollen_type,
        "forecast": {
            "Dato": [result["dato"]],
            "Pollen-nivå": [result["pollen_nivå"]],
        },
    }