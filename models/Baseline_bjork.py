import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

#Baselinemodell for Bjørk. Hvis modellen ikke er bedre enn baselinen, har den ikke lært noe.
#Baselinen gjetter at pollen i dag er lik pollen i går. 

#Laster inn pollen og værdata
df = pd.read_csv("data/full_pollen_and_weather_data_Zurich.csv", sep=";")
df["master_timestamp"] = pd.to_datetime(df["master_timestamp"])
df["dato"] = df["master_timestamp"].dt.date

#Gi nytt navn til kolonnene for å gjøre det lettere å forstå hva de representerer
df = df.rename(columns={
    "kabetuh0": "bjørk",
    "khpoach0": "gress",
    "tre200h0": "temperatur",
    "fkl010h0": "vind",
    "ure200h0": "luftfuktighet",
    "rre150h0": "nedbør",
    "sre000h0": "solskinn"
})

#Aggregere (slå sammen/oppsummer) dataene til en verdi per dag. 
daglig = df.groupby("dato").agg(
    bjørk_snitt=("bjørk", "mean"),
    bjørk_antall=("bjørk", "count"),
    temp_snitt=("temperatur", "mean"),
    temp_antall=("temperatur", "count"),
    luftfukt_snitt=("luftfuktighet", "mean"),
    luftfukt_antall=("luftfuktighet", "count"),
    solskinn_sum=("solskinn", "sum"),
    solskinn_antall=("solskinn", "count")
).reset_index()

daglig["dato"] = pd.to_datetime(daglig["dato"])

#Fjerne dager med mindre enn 20 gyldige timesmålinger for hver variabel
MIN_MALINGER = 20
gyldig = (
    (daglig["bjørk_antall"] >= MIN_MALINGER) &
    (daglig["temp_antall"] >= MIN_MALINGER) &
    (daglig["luftfukt_antall"] >= MIN_MALINGER) &
    (daglig["solskinn_antall"] >= MIN_MALINGER)
)
print(f"Antall gyldige dager etter filtrering: {gyldig.sum()} av {len(daglig)}")

#Ugyldige dager blir gjort om til NaN i stedet for at de fjernes, slik at "i går" alltid er den faktiske forrige kalenderdagen
verdi_kolonner = ["bjørk_snitt", "temp_snitt", "luftfukt_snitt", "solskinn_sum"]
daglig.loc[~gyldig, verdi_kolonner] = np.nan
daglig = daglig.set_index("dato")[verdi_kolonner].asfreq("D").reset_index()

#Gårsdagen pollen 
daglig = daglig.sort_values("dato").reset_index(drop=True)
daglig["bjørk_i_går"] = daglig["bjørk_snitt"].shift(1)

#Feautures som skal brukes
features = ["temp_snitt", "luftfukt_snitt", "solskinn_sum", "bjørk_i_går"]

#Target = dagens pollennivåer
target = "bjørk_snitt"

#Fjerner dager der noe mangler. 
daglig = daglig.dropna(subset=features + [target]).reset_index(drop=True)

#Bjørk sesong er fra mars til mai. Vi ønsker å predikere pollennivåene i denne perioden, så vi filtrerer ut alle andre måneder.
bjørk_sesong = daglig[daglig["dato"].dt.month.between(3, 5)]

#Del opp i trening-, validering- og testsett. Dette deles opp etter år
bjørk_sesong["år"] = bjørk_sesong["dato"].dt.year

trening = bjørk_sesong[bjørk_sesong["år"] == 2023]
validering = bjørk_sesong[bjørk_sesong["år"] == 2024]
test = bjørk_sesong[bjørk_sesong["år"] == 2025]

y_validering = validering[target]
y_test = test[target]

print(f"Trening: {len(trening)} dager , Validering: {len(validering)} dager , Test: {len(test)} dager")
print("Snitt pollen i valideringen:", y_validering.mean())

#Baseline gjetter at pollennivået i dag er det samme som i går. Dette er en enkel baseline som vi kan sammenligne med modellen vår for å se om modellen faktisk gir bedre prediksjoner enn bare å anta at pollennivået er det samme som dagen før.
validering_prediksjoner = validering["bjørk_i_går"]

print(f"\n--- Baseline (gjetter gårsdagen), validering (2024) ---")
print("MAE:", mean_absolute_error(y_validering, validering_prediksjoner))
print("R2:", r2_score(y_validering, validering_prediksjoner))
print("RMSE:", root_mean_squared_error(y_validering, validering_prediksjoner))

#Baseline for testsettet (2025)
test_prediksjoner = test["bjørk_i_går"]
print(f"\n--- Baseline (gjetter gårsdagen), test (2025) ---")
print("MAE:", mean_absolute_error(y_test, test_prediksjoner))
print("R2:", r2_score(y_test, test_prediksjoner))
print("RMSE:", root_mean_squared_error(y_test, test_prediksjoner))