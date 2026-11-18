import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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

#Aggregere (slå sammen/oppsummer) dataene til en verdi per dag. Dette gjør vi fordi vi ønsker å predikere daglige pollenverdier, ikke timeverdier.
daglig = df.groupby("dato").agg(
    gress_snitt=("gress", "mean"),
    gress_antall=("gress", "count"),
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
    (daglig["gress_antall"] >= MIN_MALINGER) &
    (daglig["temp_antall"] >= MIN_MALINGER) &
    (daglig["luftfukt_antall"] >= MIN_MALINGER) &
    (daglig["solskinn_antall"] >= MIN_MALINGER)
)
print(f"Antall gyldige dager etter filtrering: {gyldig.sum()} av {len(daglig)}")

#Ugyldige dager blir gjort om til NaN i stedet for at de fjernes, slik at "i går" alltid er den faktiske forrige kalenderdagen
verdi_kolonner = ["gress_snitt", "temp_snitt", "luftfukt_snitt", "solskinn_sum"]
daglig.loc[~gyldig, verdi_kolonner] = np.nan
daglig = daglig.set_index("dato")[verdi_kolonner].asfreq("D").reset_index()

#Gårsdagen pollen 
daglig = daglig.sort_values("dato").reset_index(drop=True)
daglig["gress_i_går"] = daglig["gress_snitt"].shift(1)

#Feautures som skal brukes
features = ["temp_snitt", "luftfukt_snitt",  "solskinn_sum", "gress_i_går"]

#Target = dagens pollennivåer
target = "gress_snitt"

#Fjerner dager der noe mangler. GJør dette fordi LinearRegression ikke kan håndtere NaN-verdier. Tidligere i koden endret vi ugyldige dager til Nan, men dette var bare for at det skulle gi riktig "i går", men vi trenger ikke de til noe annet. 
daglig = daglig.dropna(subset=features + [target]).reset_index(drop=True)

#Gress sesong er fra april til september. Vi ønsker å predikere pollennivåene i denne perioden, så vi filtrerer ut alle andre måneder.
gress_sesong = daglig[daglig["dato"].dt.month.between(4, 9)]

#Del opp i trening-, validering- og testsett. Dette deles opp etter år
gress_sesong["år"] = gress_sesong["dato"].dt.year

trening = gress_sesong[gress_sesong["år"].isin([2023, 2024])]
validering = gress_sesong[gress_sesong["år"] == 2025]
test = gress_sesong[gress_sesong["år"] == 2026]

y_validering = validering[target]
y_test = test[target]

print(f"Trening: {len(trening)} dager , Validering: {len(validering)} dager , Test: {len(test)} dager")
print("Snitt pollen i valideringen:", y_validering.mean())

#Baseline gjetter at pollennivået i dag er det samme som i går. Dette er en enkel baseline som vi kan sammenligne med modellen vår for å se om modellen faktisk gir bedre prediksjoner enn bare å anta at pollennivået er det samme som dagen før.
validering_prediksjoner = validering["gress_i_går"]

print(f"\n--- Baseline (gjetter gårsdagen), validering (2025) ---")
print("MAE:", mean_absolute_error(y_validering, validering_prediksjoner))
print("R2:", r2_score(y_validering, validering_prediksjoner))
print("RMSE:", root_mean_squared_error(y_validering, validering_prediksjoner))

#Baseline for testsettet (2026)
test_prediksjoner = test["gress_i_går"]
print(f"\n--- Baseline (gjetter gårsdagen), test (2026) ---")
print("MAE:", mean_absolute_error(y_test, test_prediksjoner))
print("R2:", r2_score(y_test, test_prediksjoner))
print("RMSE:", root_mean_squared_error(y_test, test_prediksjoner))