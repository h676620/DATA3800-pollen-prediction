import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


############################################################
# Leser inn og klargjør datasettet
############################################################

# Leser inn datasettet
df = pd.read_csv(
    "data/full_pollen_and_weather_data_Zurich.csv",
    sep=";"
)

# Gir kolonnene nye navn for å gjøre det mer forståelig
df = df.rename(columns={
    "station_abbr_pollen": "Pollenstasjon",
    "station_abbr_weather": "Værstasjon",
    "master_timestamp": "Tid",
    "timestamp_pollen": "Pollen_tid",
    "timestamp_weather": "Vær_tid",

    "kabetuh0": "Bjørk",
    "khpoach0": "Gress",

    "tre200h0": "Lufttemperatur",
    "fkl010h0": "Vindstyrke",
    "ure200h0": "Luftfuktighet",
    "rre150h0": "Nedbør",

    "sre000h0": "Solskinn",
    "dkl010h0": "Vindretning"
})


# Velger ut kolonnene som er relevante
data = df[
    [
        "Tid",
        "Bjørk",
        "Lufttemperatur",
        "Vindstyrke",
        "Luftfuktighet",
        "Solskinn",
        "Nedbør"
    ]
].copy()


# Konverterer tidskolonnen til datetime
data["Tid"] = pd.to_datetime(data["Tid"])

# Lager dato uten klokkeslett
data["dato"] = data["Tid"].dt.date


############################################################
# Gjør timesdata om til daglige verdier
############################################################

# Aggregere dataene til én verdi per dag.
# Vi har også med antall timesmålinger hver dag bygger på,
# slik at vi senere kan kontrollere at hver dag har minst
# 20 gyldige målinger.
daglig_alle = data.groupby("dato").agg(

    bjørk_mean=("Bjørk", "mean"),
    bjørk_antall=("Bjørk", "count"),

    temperatur_mean=("Lufttemperatur", "mean"),
    temperatur_antall=("Lufttemperatur", "count"),

    vind_mean=("Vindstyrke", "mean"),
    vind_antall=("Vindstyrke", "count"),

    luftfuktighet_mean=("Luftfuktighet", "mean"),
    luftfuktighet_antall=("Luftfuktighet", "count"),

    solskinn_total=("Solskinn", "sum"),
    solskinn_antall=("Solskinn", "count"),

    nedbør_total=("Nedbør", "sum"),
    nedbør_antall=("Nedbør", "count")

).reset_index()


# Gjør dato til datetime
daglig_alle["dato"] = pd.to_datetime(daglig_alle["dato"])


# Lager kolonnen dag_i_året,
# slik at modellen kan vite hvor i året/pollensesongen vi er
daglig_alle["dag_i_året"] = daglig_alle["dato"].dt.dayofyear


############################################################
# Datarensing
############################################################

# Beholder bare dager som har minst 20 gyldige
# timesmålinger for alle variablene
daglig_gyldige = daglig_alle[
    (daglig_alle["bjørk_antall"] >= 20) &
    (daglig_alle["temperatur_antall"] >= 20) &
    (daglig_alle["vind_antall"] >= 20) &
    (daglig_alle["luftfuktighet_antall"] >= 20) &
    (daglig_alle["solskinn_antall"] >= 20) &
    (daglig_alle["nedbør_antall"] >= 20)
].copy()


# Sorterer datoene kronologisk
daglig_gyldige = daglig_gyldige.sort_values("dato").copy()


# Printer informasjon om datasettet etter datarensing
print("\nStørrelse etter datarensing:")
print(daglig_gyldige.shape)

print("\nFørste rader:")
print(daglig_gyldige.head())

print("\nStatistikk:")
print(daglig_gyldige.describe())


############################################################
# Feature engineering: bjørkepollen dagen før
############################################################

# Lager ny kolonne med bjørkepollenkonsentrasjonen dagen før
daglig_gyldige["bjørk_i_går"] = (
    daglig_gyldige["bjørk_mean"].shift(1)
)


# Lager også en kolonne med datoen til forrige rad
# for å kontrollere at verdien faktisk kommer fra dagen før
daglig_gyldige["i_går"] = (
    daglig_gyldige["dato"].shift(1)
)


# Beholder bare rader hvor forrige observasjon faktisk
# kommer fra kalenderdagen før
modell_data = daglig_gyldige[
    daglig_gyldige["i_går"]
    ==
    daglig_gyldige["dato"] - pd.Timedelta(days=1)
].copy()


# Kolonnen i_går var bare nødvendig for kontrollen,
# så den kan fjernes nå
modell_data = modell_data.drop(columns=["i_går"])


print("\nModell-data:")
print(modell_data.head())
print(modell_data.shape)


############################################################
# Beholder bjørkepollensesongen
############################################################

# Bjørkepollensesong: mars til mai
bjork_sesong = modell_data[
    modell_data["dato"].dt.month.between(3, 5)
].copy()


############################################################
# XGBoost-modell
############################################################

# Features/inputs modellen skal bruke
features = [
  "bjørk_i_går",
  "temperatur_mean",
  "luftfuktighet_mean",
  "solskinn_total"
]


############################################################
# Train / validation / test split
############################################################

# Train: 2023
train_data = bjork_sesong[
    bjork_sesong["dato"].dt.year == 2023
].copy()

# Validation: 2024
val_data = bjork_sesong[
    bjork_sesong["dato"].dt.year == 2024
].copy()

# Test: 2025
test_data = bjork_sesong[
    bjork_sesong["dato"].dt.year == 2025
].copy()


# Input X og target y for treningsdata
X_train = train_data[features]
y_train = train_data["bjørk_mean"]

# Input X og target y for valideringsdata
X_val = val_data[features]
y_val = val_data["bjørk_mean"]

# Input X og target y for testdata
X_test = test_data[features]
y_test = test_data["bjørk_mean"]


print("\nDatasplit:")
print("Trening 2023:", len(train_data), "dager")
print("Validering 2024:", len(val_data), "dager")
print("Test 2025:", len(test_data), "dager")


############################################################
# Oppretter og trener modellen
############################################################

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)


# Trener modellen kun på 2023
model.fit(X_train, y_train)


############################################################
# Validering på 2024
############################################################

# Lager prediksjoner på valideringsdata
y_val_pred = model.predict(X_val)


# Pollenkonsentrasjon kan ikke være negativ.
# Alle negative prediksjoner settes derfor til 0.
y_val_pred = np.maximum(y_val_pred, 0)


############################################################
# Evaluerer modellen på valideringsdata
############################################################

# MAE = hvor mye modellen bommer i gjennomsnitt
val_mae = mean_absolute_error(
    y_val,
    y_val_pred
)

# RMSE gir større straff til store prediksjonsfeil
val_rmse = root_mean_squared_error(
    y_val,
    y_val_pred
)

# R² viser hvor mye av variasjonen i pollen
# modellen klarer å forklare
val_r2 = r2_score(
    y_val,
    y_val_pred
)


print("\nVALIDATION – 2024")
print("Validation MAE:", val_mae)
print("Validation RMSE:", val_rmse)
print("Validation R^2:", val_r2)


############################################################
# Plotter faktisk vs. predikert pollen
############################################################

plt.figure(figsize=(12, 6))


# Faktiske pollenverdier i 2024
plt.plot(
    val_data["dato"],
    y_val,
    label="Faktisk pollen"
)


# Modellens prediksjoner i 2024
plt.plot(
    val_data["dato"],
    y_val_pred,
    label="Predikert pollen"
)


plt.xlabel("Dato")
plt.ylabel("Bjørkepollenkonsentrasjon")
plt.title(
    "Validering 2024: faktisk vs. predikert bjørkepollen"
)

plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()