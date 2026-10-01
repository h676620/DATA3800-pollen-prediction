import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Laster inn pollen og værdata
df = pd.read_csv("data/full_pollen_and_weather_data_Zurich.csv", sep=";")
df["master_timestamp"] = pd.to_datetime(df["master_timestamp"])
df["dato"] = df["master_timestamp"].dt.date

#Aggregere (slå sammen/oppsummer) dataene til en verdi per dag. Dette gjør vi fordi vi ønsker å predikere daglige pollenverdier, ikke timeverdier.
#Vi ønsker å finne gjennomsnittet for bjørkepollen, temp, vind og luftfuktighet
#Vi ønsker å finne summen for nedbør
#Vi ønsker å telle antall gyldige målinger per deg for hver variabel
daglig = df.groupby("dato").agg(
    bjork_snitt=("kabetuh0", "mean"),
    bjork_antall=("kabetuh0", "count"),
    temp_snitt=("tre200h0", "mean"),
    temp_antall=("tre200h0", "count"),
    vind_snitt=("fkl010h0", "mean"),
    vind_antall=("fkl010h0", "count"),
    luftfukt_snitt=("ure200h0", "mean"),
    luftfukt_antall=("ure200h0", "count"),
    nedbor_sum=("rre150h0", "sum"),
    nedbor_antall=("rre150h0", "count"),
).reset_index()

daglig["dato"] = pd.to_datetime(daglig["dato"])

#Fjerne dager med mindre enn 20 gyldige timesmålinger for hver variabel
MIN_MALINGER = 20
gyldig = (
    (daglig["bjork_antall"] >= MIN_MALINGER) &
    (daglig["temp_antall"] >= MIN_MALINGER) &
    (daglig["vind_antall"] >= MIN_MALINGER) &
    (daglig["luftfukt_antall"] >= MIN_MALINGER) &
    (daglig["nedbor_antall"] >= MIN_MALINGER)
)
daglig = daglig[gyldig].reset_index(drop=True)
print(f"Antall gyldige dager etter filtrering: {len(daglig)}")

#Legg til en kolonne for dag i året. Dette gir hver dag et nummer (altså 1-365) som vi kan bruke som en prediktor i modellen. Dette er nyttig fordi pollenmengden ofte varierer gjennom året.
daglig["dag_i_aret"] = daglig["dato"].dt.dayofyear

#Definere target, det vil være å predikere morgendagens bjørkpollen

#Den siste dagen i datasettet har ingen "i morgen", så den må fjernes

#Del opp i trening-, validering- og testsett. Dette deles opp etter år

#Tren Linar regresjonsmodell med treningssettet

#Evaluer den linear regresjonsmodellen på valideringssettet. Her kan bruker vi  R2 score og mean absolute error (MAE) 