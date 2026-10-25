import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
#Vi ønsker å finne gjennomsnittet for gresspollen, temp, vind og luftfuktighet
#Vi ønsker å finne summen for nedbør
#Vi ønsker å telle antall gyldige målinger per dag for hver variabel
daglig = df.groupby("dato").agg(
    gress_snitt=("gress", "mean"),
    gress_antall=("gress", "count"),
    temp_snitt=("temperatur", "mean"),
    temp_antall=("temperatur", "count"),
    vind_snitt=("vind", "mean"),
    vind_antall=("vind", "count"),
    luftfukt_snitt=("luftfuktighet", "mean"),
    luftfukt_antall=("luftfuktighet", "count"),
    nedbør_sum=("nedbør", "sum"),
    nedbør_antall=("nedbør", "count"),
    solskinn_sum=("solskinn", "sum"),
    solskinn_antall=("solskinn", "count")
).reset_index()

daglig["dato"] = pd.to_datetime(daglig["dato"])

#Fjerne dager med mindre enn 20 gyldige timesmålinger for hver variabel
MIN_MALINGER = 20
gyldig = (
    (daglig["gress_antall"] >= MIN_MALINGER) &
    (daglig["temp_antall"] >= MIN_MALINGER) &
    (daglig["vind_antall"] >= MIN_MALINGER) &
    (daglig["luftfukt_antall"] >= MIN_MALINGER) &
    (daglig["nedbør_antall"] >= MIN_MALINGER) &
    (daglig["solskinn_antall"] >= MIN_MALINGER)
)
print(f"Antall gyldige dager etter filtrering: {gyldig.sum()} av {len(daglig)}")

#Ugyldige dager blir gjort om til NaN i stedet for at de fjernes, slik at "i går" alltid er den faktiske forrige kalenderdagen
verdi_kolonner = ["gress_snitt", "temp_snitt", "vind_snitt", "luftfukt_snitt", "nedbør_sum", "solskinn_sum"]
daglig.loc[~gyldig, verdi_kolonner] = np.nan
daglig = daglig.set_index("dato")[verdi_kolonner].asfreq("D").reset_index()

#Legg til en kolonne for dag i året. Dette gir hver dag et nummer (altså 1-365) som vi kan bruke som en prediktor i modellen. Dette er nyttig fordi pollenmengden ofte varierer gjennom året.
daglig["dag_i_året"] = daglig["dato"].dt.dayofyear

#Gårsdagen pollen 
daglig = daglig.sort_values("dato").reset_index(drop=True)
daglig["gress_i_går"] = daglig["gress_snitt"].shift(1)

#Feautures som skal brukes
features = ["dag_i_året", "temp_snitt", "vind_snitt", "luftfukt_snitt", "nedbør_sum", "solskinn_sum", "gress_i_går"]

#Target = dagens pollennivåer
target = "gress_snitt"

#Fjerner dager der noe mangler. GJør dette fordi LinearRegression ikke kan håndtere NaN-verdier. Tidligere i koden endret vi ugyldige dager til Nan, men dette var bare for at det skulle gi riktig "i går", men vi trenger ikke de til noe annet. 
daglig = daglig.dropna(subset=features + [target]).reset_index(drop=True)

#Gress sesong er april til september. Vi ønsker å predikere pollennivåene i denne perioden, så vi filtrerer ut alle andre måneder.
gress_sesong = daglig[daglig["dato"].dt.month.between(4, 9)]

#Del opp i trening-, validering- og testsett. Dette deles opp etter år
gress_sesong["år"] = gress_sesong["dato"].dt.year

trening = gress_sesong[gress_sesong["år"].isin([2023, 2024])]
validering = gress_sesong[gress_sesong["år"] == 2025]
test = gress_sesong[gress_sesong["år"] == 2026]

X_trening, y_trening = trening[features], trening[target]
X_validering, y_validering = validering[features], validering[target]
X_test, y_test = test[features], test[target]

print(f"Trening: {len(X_trening)} dager, Validering: {len(X_validering)} dager, Test: {len(X_test)} dager")

#Tren Linar regresjonsmodell med treningssettet
modell = LinearRegression()
modell.fit(X_trening, y_trening)

#Evaluer den linear regresjonsmodellen med valideringssettet. Her  bruker vi  R2 score og MAE (mean absolute error)
# R2 score er et mål på hvor godt modellen forklarer variasjonen i dataene. En R2 score på 1 betyr at modellen forklarer all variasjon, mens en R2 score på 0 betyr at modellen ikke forklarer noen variasjon.
# MAE er et mål på hvor langt unna modellens prediksjoner er fra de faktiske verdiene. En lav MAE betyr at modellen predikerer godt, mens en høy MAE betyr at modellen predikerer dårlig. 
validering_prediksjoner = modell.predict(X_validering)
print("\n--- Validering (2025) ---")
print("MAE:", mean_absolute_error(y_validering, validering_prediksjoner))
print("R2:", r2_score(y_validering, validering_prediksjoner))
print("RMSE:", root_mean_squared_error(y_validering, validering_prediksjoner))

#Snitt pollen i valideringperioden, for å vurdere hvor stor MAE er i forhold til gjennomsnittlig pollennivå. Dette gir oss en bedre forståelse av hvor godt modellen predikerer pollennivåene.
print("Snitt pollen i valideringen:", y_validering.mean())

#Sammenligningsmodell (baseline) som predikerer at dagens pollennivå er det samme som gårsdagens pollennivå. Dette er en enkel modell som vi kan bruke for å vurdere hvor mye bedre vår lineære regresjonsmodell er.
print("MAE hvis vi finner ut av gårsdagens pollennivå", mean_absolute_error(y_validering, X_validering["gress_i_går"]))

#Tren modellen på både trenings- og valideringssettet.
X_final = pd.concat([X_trening, X_validering])
y_final = pd.concat([y_trening, y_validering])
final_modell = LinearRegression()
final_modell.fit(X_final, y_final)

# Evaluer deretter med testsettet. Skal gjøres til slutt, kun en gang. 
#test_prediksjoner = final_modell.predict(X_test)
#print("\n--- Test (2026) ---")
#print("MAE:", mean_absolute_error(y_test, test_prediksjoner))
#print("R2:", r2_score(y_test, test_prediksjoner))

#Graf
#import matplotlib.pyplot as plt

#plt.figure(figsize=(12, 5))
#plt.plot(test["dato"], y_test, label="Faktisk pollen")
#plt.plot(test["dato"], test_prediksjoner, label="Predikert pollen")
#plt.xlabel("Dato")
#plt.ylabel("Gresspollen")
#plt.title("Pollenprediksjon")
#plt.legend()
#plt.tight_layout()
#plt.savefig("Linear_regression_gress.png", dpi=150)
#plt.show()

#Koefissientene
print("\n--- Koefissienter ---")
for navn, koeff in zip(features, final_modell.coef_):
    print(f"{navn}: {koeff: .3f}")
print(f"Konstantledd: {final_modell.intercept_: .3f}")
