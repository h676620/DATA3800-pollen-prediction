from datetime import datetime
from pathlib import Path

import pandas as pd


CSV_FILES = {
    "bjørk": "predikert_bjork.csv",
    "gress": "predikert_gress.csv",
}


def lookup_pollen(pollen_type: str, today_date: str) -> dict:
    pollen_type = pollen_type.strip().casefold()

    if pollen_type not in CSV_FILES:
        raise ValueError("Pollentype må være 'bjørk' eller 'gress'.")

    wanted_date = datetime.strptime(today_date, "%d.%m.%Y").date()
    csv_path = Path(__file__).with_name(CSV_FILES[pollen_type])
    table = pd.read_csv(csv_path)

    dates = pd.to_datetime(
        table["dato"],
        dayfirst=True,
        errors="coerce",
    ).dt.date

    matches = table.loc[dates == wanted_date]

    if matches.empty:
        level = None
    else:
        level = matches.iloc[0]["predikert_nivå"]
        if pd.isna(level):
            level = None

    return {
        "dato": today_date,
        "pollen_nivå": level,
    }