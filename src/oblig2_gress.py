import pandas as pd
import matplotlib.pyplot as plt

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

# Lager en kolonne dato uten klokkeslett
data["dato"] = data["Tid"].dt.date

# Gjør timesdata om til daglige verdier, og har med antall timesmålinger hver dag bygger på 
# for å senere sjekke at hver dag har minst 20 gyldige timer/målinger.
# Gjennomsnitt brukes for pollen, temperatur, vind og luftfuktighet. Sum brukes for solskinn og nedbør.
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
).reset_index() #gjør "dato" om til en vanlig kolonne igjen


# Gjør dato-kolonnen til datetime
daglig_alle["dato"] = pd.to_datetime(daglig_alle["dato"])

#lager kolonnen "dag_i_året" som forteller oss hvor i året vi er
daglig_alle["dag_i_året"] = daglig_alle["dato"].dt.dayofyear


# Bruker bare de dagene/radene som har minst 20 gyldige timer/målinger. Filtrerer ut de dagene som har mindre enn 20 gyldige målinger for noen av variablene.
daglig_gyldige = daglig_alle[
    (daglig_alle["gress_antall"] >= 20) &
    (daglig_alle["temperatur_antall"] >= 20) &
    (daglig_alle["vind_antall"] >= 20) &
    (daglig_alle["luftfuktighet_antall"] >= 20) &
    (daglig_alle["solskinn_antall"] >= 20) &
    (daglig_alle["nedbør_antall"] >= 20)
].copy()

daglig_gyldige = daglig_gyldige.sort_values("dato").copy()

print("Før filtrering:", daglig_alle.shape)
print("Etter filtrering:", daglig_gyldige.shape)

# Sorterer datoene kronologisk (trengs kanskje egentlig ikke, bare for sikkerhet, kan droppe)
daglig_gyldige = daglig_gyldige.sort_values("dato").copy()

# for å se hvordan datasettet ser ut etter datarensing + kontroll av manglende verdier
print("\nStørrelse etter datarensing:")
print(daglig_gyldige.shape)

print(daglig_gyldige.head())

print("\nManglende verdier etter datarensing:")
print(daglig_gyldige.isna().sum())

print("\nStatistikk:")
print(daglig_gyldige.describe())

# plotter graf for å se gresspollen over tid

plt.figure(figsize=(12, 6))

plt.plot(
    daglig_gyldige["dato"],
    daglig_gyldige["gress_mean"]
)

plt.xlabel("Dato")
plt.ylabel("Gjennomsnittlig gresspollenkonsentrasjon")
plt.title("Daglig gresspollenkonsentrasjon i Zürich")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# plotte bare et spesifikt år for å se hvordan pollensesongen ser ut
# 2024:
data_2024 = daglig_gyldige[
    daglig_gyldige["dato"].dt.year == 2024
].copy()

# Plotter gresspollen for 2024
plt.figure(figsize=(12, 6))

plt.plot(
    data_2024["dato"],
    data_2024["gress_mean"]
)

plt.xlabel("Dato")
plt.ylabel("Gjennomsnittlig gresspollenkonsentrasjon")
plt.title("Gresspollen i Zürich - 2024")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()