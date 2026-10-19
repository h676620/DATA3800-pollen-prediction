import pandas as pd

df = pd.read_csv(
    "data/full_pollen_and_weather_data_Zurich.csv",
    sep=";"
)

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

# Gjør tidskolonnen om til datetime
data["Tid"] = pd.to_datetime(data["Tid"])

# Lager dato uten klokkeslett
data["dato"] = data["Tid"].dt.date

# Gjør timesdata om til daglige verdier, og har med antall timesmålinger hver dag bygger på for å senere sjekke at hver dag har minst 20 gyldige timer/målinger.
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
).reset_index() #gjør "dato" om til en vanlig kolonne igjen


# Gjør "dato" til datetime
daglig_alle["dato"] = pd.to_datetime(daglig_alle["dato"])

#lager kolonnen "dag_i_året" slik at modellen kan vite hvor i pollensesongen vi er
daglig_alle["dag_i_året"] = daglig_alle["dato"].dt.dayofyear


# Bruker bare de dagene/radene som har minst 20 gyldige timer/målinger.
daglig_gyldige = daglig_alle[
    (daglig_alle["bjørk_antall"] >= 20) &
    (daglig_alle["temperatur_antall"] >= 20) &
    (daglig_alle["vind_antall"] >= 20) &
    (daglig_alle["luftfuktighet_antall"] >= 20) &
    (daglig_alle["solskinn_antall"] >= 20) &
    (daglig_alle["nedbør_antall"] >= 20)
].copy()

# Sorterer datoene kronologisk før vi lager target (trengs egentlig ikke, bare for sikkerhet, kan droppe)
daglig_gyldige = daglig_gyldige.sort_values("dato").copy()

# Lager neste dato og target for neste dag
daglig_gyldige["neste_dag"] = daglig_gyldige["dato"].shift(-1)
daglig_gyldige["bjørk_i_morgen"] = daglig_gyldige["bjørk_mean"].shift(-1)

# Det mangler en del dager, så vi må passe på "hullene": 
# vi beholder bare rader der neste gyldige observasjon faktisk er dagen etter
modell_data = daglig_gyldige[
    daglig_gyldige["neste_dag"]
    == daglig_gyldige["dato"] + pd.Timedelta(days=1)
].copy()

modell_data = modell_data.drop(columns=["neste_dag"])

print(modell_data.head())
print(modell_data.shape)

# Begynner med XGBoost:
features = [
    "bjørk_mean",
    "temperatur_mean",
    "vind_mean",
    "luftfuktighet_mean",
    "solskinn_total",
    "nedbør_total",
    "dag_i_året"
]

# lager datasett for treningsdata (2023 og 2024)
train_data = modell_data[
    modell_data["dato"].dt.year < 2025
].copy()

# lager datasett for testdata (2025)
test_data = modell_data[
    modell_data["dato"].dt.year == 2025
].copy()

# input og target til treningsdata
X_train = train_data[features]
y_train = train_data["bjørk_i_morgen"]

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)

# trener modellen
model.fit(X_train, y_train)

# input og target til testdata
X_test = test_data[features]
y_test = test_data["bjørk_i_morgen"]

# tester modellen
y_pred = model.predict(X_test)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

##############

# plotter resultatet:

import matplotlib.pyplot as plt

# størrelse på figuren
plt.figure(figsize=(12, 6))

# denne grafen hviser faktisk pollenkonsentrasjon
# x-aksen er dato i 2025, y-aksen er faktisk bjørkepollen-konnsentrasjon i 2025
plt.plot(
    test_data["dato"],
    y_test,
    label="Faktisk pollen"
)


# denne grafen viser predikert pollen-konsentrasjon
# x-aksen er dato i 2025, y-aksen er predikert bjørkepollen-konnsentrasjon i 2025
plt.plot(
    test_data["dato"],
    y_pred,
    label="Predikert pollen"
)

# navngir x-aksen
plt.xlabel("Dato")

# navngir y-aksen
plt.ylabel("Bjørkepollenkonsentrasjon")

# navngir figuren
plt.title("Faktisk vs. predikert bjørkepollen i 2025")

# navngir/forklarer linjene
plt.legend()

# viser grafen
plt.show()

#######################

# evaluering
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# sammenligner predikert pollenverdier og faktisk pollenverdier
# mae er hvor mye modellen bommer i gjennomsnitt
mae = mean_absolute_error(y_test, y_pred)  

# rmse er også gjennomsnitt bare squared, så store bomminger/prediksjonsfeil gir større rmse-tall
rmse = root_mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)