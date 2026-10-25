import pandas as pd

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

# Velger ut de kolonnene som er relevante
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


# Bruker bare de dagene/radene som har minst 20 gyldige timer/målinger. Filtrerer ut de dagene som har mindre enn 20 gyldige målinger for noen av variablene.
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

# enkel print for å se hvordan datasettet ser ut etter datarensing
print("\nStørrelse etter datarensing:")
print(daglig_gyldige.shape)

print(daglig_gyldige.head())

print("\nstatistikk:")
print(daglig_gyldige.describe())

# plotte graf for å se hvordan datasettet ser ut etter datarensing...:


# lager ny kolonne som er hva bjørkekonsentrasjonen var dagen før, slik at modellen kan bruke det som input for å predikere konsentrasjonen i dag
daglig_gyldige["bjørk_i_går"] = daglig_gyldige["bjørk_mean"].shift(+1)

# må sørge for at "bjørk_i_går" faktisk kommer fra dagen før (dato-1) og ikke fra flere dager siden, siden vi har dager/rader som mangler:
# lager kolonne som er datoen forrige dag 
daglig_gyldige["i_går"] = daglig_gyldige["dato"].shift(+1)

# lager nytt datasett der "bjørk_i_går" faktisk kommer fra dagen før
modell_data = daglig_gyldige[daglig_gyldige["i_går"] == daglig_gyldige["dato"] + pd.Timedelta(days=-1)].copy()

# nå kan vi droppe kolonnen "i_går" som vi bare brukte for å sjekke at "bjørk_i_går" faktisk kom fra dagen før
modell_data = modell_data.drop(columns=["i_går"])

print(modell_data.head())
print(modell_data.shape)

# Beholder bare bjørkepollensesongen: 01.mars - 30.april
bjork_sesong = modell_data[
    modell_data["dato"].dt.month.between(3, 4)
].copy()


############################################################
# Starter med XGBoost-modellen:
############################################################

# Velger ut features/inputs for XGBoost
features = [
    "bjørk_i_går",
    "temperatur_mean",
    "vind_mean",
    "luftfuktighet_mean",
    "solskinn_total",
    "nedbør_total",
    "dag_i_året"
]


# Deler opp datasettet i treningsdata, valideringsdata og testdata basert på årstall. 
# Vi bruker 2023 til trening og 2024 til validering først, og så etter valideringen bruker vi både 2023 og 2024 som treningsdata og 2025 som final test:

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


# input(x) og target(y) til treningsdata
X_train = train_data[features]
y_train = train_data["bjørk_mean"]

# input(x) og target(y) til valideringsdata
X_val = val_data[features]
y_val = val_data["bjørk_mean"]

# input(x) og target(y) til testdata
X_test = test_data[features]
y_test = test_data["bjørk_mean"]


from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)

# trener modellen (bare på 2023)
model.fit(X_train, y_train)

# tester modellen på valideringsdata (2024)
y_val_pred = model.predict(X_val)

# evaluerer modellen på valideringsdata (2024)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# sammenligner predikert pollenverdier og faktisk pollenverdier:

# mae er hvor mye modellen bommer i gjennomsnitt
val_mae = mean_absolute_error(y_val, y_val_pred)
# rmse er også gjennomsnitt bare squared, så store bomminger/prediksjonsfeil gir større rmse-tall
val_rmse = root_mean_squared_error(y_val, y_val_pred)

print("Validation MAE:", val_mae) 
print("Validation RMSE:", val_rmse)


##############
# plotter resultatet:
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Faktiske pollenverdier i 2024
plt.plot(
    val_data["dato"],
    y_val,
    label="Faktisk pollen"
)

# Modellens prediksjoner for 2024
plt.plot(
    val_data["dato"],
    y_val_pred,
    label="Predikert pollen"
)

plt.xlabel("Dato")
plt.ylabel("Bjørkepollenkonsentrasjon")
plt.title("Validering 2024: faktisk vs. predikert bjørkepollen")

plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


