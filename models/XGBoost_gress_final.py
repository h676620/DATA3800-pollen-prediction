import pandas as pd

# Leser inn datasettet
df = pd.read_csv(
    "data/full_pollen_and_weather_data_Zurich.csv",
    sep=";"
)

# Gir kolonnene mer forståelige navn
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

# Velger ut relevante kolonner
data = df[
    [
        "Tid",
        "Gress",
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


# Gjør timesdata om til daglige verdier
daglig_alle = data.groupby("dato").agg(
    gress_mean=("Gress", "mean"),
    gress_antall=("Gress", "count"),

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

# Lager dag_i_året.
# (brukes ikke i final-modellen, men kan beholdes i datasettet)
daglig_alle["dag_i_året"] = daglig_alle["dato"].dt.dayofyear


# Filtrerer dager med for få målinger
daglig_gyldige = daglig_alle[
    (daglig_alle["gress_antall"] >= 20) &
    (daglig_alle["temperatur_antall"] >= 20) &
    (daglig_alle["vind_antall"] >= 20) &
    (daglig_alle["luftfuktighet_antall"] >= 20) &
    (daglig_alle["solskinn_antall"] >= 20) &
    (daglig_alle["nedbør_antall"] >= 20)
].copy()

# Sorterer kronologisk
daglig_gyldige = daglig_gyldige.sort_values("dato").copy()

print("\nStørrelse etter datarensing:")
print(daglig_gyldige.shape)

print("\nStatistikk:")
print(daglig_gyldige.describe())


# Lager feature: gresspollen dagen før
daglig_gyldige["gress_i_går"] = daglig_gyldige["gress_mean"].shift(1)

# Lagrer datoen til forrige rad
daglig_gyldige["i_går"] = daglig_gyldige["dato"].shift(1)

# Beholder bare rader hvor forrige rad faktisk er dagen før
modell_data = daglig_gyldige[
    daglig_gyldige["i_går"]
    == daglig_gyldige["dato"] - pd.Timedelta(days=1)
].copy()

# Trenger ikke denne kolonnen videre
modell_data = modell_data.drop(columns=["i_går"])


# Beholder gresspollensesongen: mai - august: ny (april-september)
gress_sesong = modell_data[
    modell_data["dato"].dt.month.between(4, 9)
].copy()


############################################################
# Final XGBoost-modell
############################################################

# Features valgt på validation-settet (2025)
features = [
    "gress_i_går",
    "luftfuktighet_mean",
    "solskinn_total",
    "nedbør_total"
]


############################################################
# Train: 2023 + 2024 + 2025
# Final test: 2026
############################################################

train_data = gress_sesong[
    gress_sesong["dato"].dt.year.isin([2023, 2024, 2025])
].copy()

test_data = gress_sesong[
    gress_sesong["dato"].dt.year == 2026
].copy()


# Input (X) og target (y) til treningsdata
X_train = train_data[features]
y_train = train_data["gress_mean"]

# Input (X) og target (y) til final test
X_test = test_data[features]
y_test = test_data["gress_mean"]


print("\nAntall treningsdager (2023 + 2024 + 2025):", len(train_data))
print("Antall testdager (2026):", len(test_data))


############################################################
# Trener den endelige modellen:
##########################################

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)


# Final test på 2026:
y_test_pred = model.predict(X_test)


####################################################
# Evaluerer modellen
#################################################

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print("\nFINAL TEST – 2026")
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)


###########################################################
# Plotter faktisk vs. predikert gresspollen
##########################################################

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Faktiske pollenverdier i 2026
plt.plot(
    test_data["dato"],
    y_test,
    label="Faktisk pollen"
)

# Modellens prediksjoner for 2026
plt.plot(
    test_data["dato"],
    y_test_pred,
    label="Predikert pollen"
)

plt.xlabel("Dato")
plt.ylabel("Gresspollenkonsentrasjon")
plt.title("Final test 2026: faktisk vs. predikert gresspollen")

plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()