import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, ConfusionMatrixDisplay

#Deler pollenmengden inn i tre kategorier
def pollenklasse(verdi):
    if verdi < 11:
        return "lav"
    elif verdi < 70:
        return "moderat"
    else:
        return "høy"

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
#Vi ønsker å finne gjennomsnittet for bjørkepollen, temp, vind og luftfuktighet
#Vi ønsker å finne summen for nedbør
#Vi ønsker å telle antall gyldige målinger per dag for hver variabel
daglig = df.groupby("dato").agg(
    bjørk_snitt=("bjørk", "mean"),
    bjørk_antall=("bjørk", "count"),
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

#Legg til en kolonne for dag i året. Dette gir hver dag et nummer (altså 1-365) som vi kan bruke som en prediktor i modellen. Dette er nyttig fordi pollenmengden ofte varierer gjennom året.
daglig["dag_i_året"] = daglig["dato"].dt.dayofyear

#Gårsdagen pollen 
daglig = daglig.sort_values("dato").reset_index(drop=True)
daglig["bjørk_i_går"] = daglig["bjørk_snitt"].shift(1)

#Feautures som skal brukes
features = ["temp_snitt", "luftfukt_snitt", "solskinn_sum", "bjørk_i_går"]

#Target = dagens pollennivåer
target = "bjørk_snitt"

#Fjerner dager der noe mangler. GJør dette fordi LinearRegression ikke kan håndtere NaN-verdier. Tidligere i koden endret vi ugyldige dager til Nan, men dette var bare for at det skulle gi riktig "i går", men vi trenger ikke de til noe annet. 
daglig = daglig.dropna(subset=features + [target]).reset_index(drop=True)

#Bjørk sesong er fra mars til mai. Vi ønsker å predikere pollennivåene i denne perioden, så vi filtrerer ut alle andre måneder.
bjørk_sesong = daglig[daglig["dato"].dt.month.between(3, 5)]

#Del opp i trening-, validering- og testsett. Dette deles opp etter år
bjørk_sesong["år"] = bjørk_sesong["dato"].dt.year

trening = bjørk_sesong[bjørk_sesong["år"] == 2023]
validering = bjørk_sesong[bjørk_sesong["år"] == 2024]
test = bjørk_sesong[bjørk_sesong["år"] == 2025]

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
print("\n--- Validering (2024) ---")
print("MAE:", mean_absolute_error(y_validering, validering_prediksjoner))
print("R2:", r2_score(y_validering, validering_prediksjoner))
print("RMSE:", root_mean_squared_error(y_validering, validering_prediksjoner))

#Snitt pollen i valideringperioden, for å vurdere hvor stor MAE er i forhold til gjennomsnittlig pollennivå. Dette gir oss en bedre forståelse av hvor godt modellen predikerer pollennivåene.
print("Snitt pollen i valideringen:", y_validering.mean())
                                                            

#Tren modellen på både trenings- og valideringssettet.
X_final = pd.concat([X_trening, X_validering])
y_final = pd.concat([y_trening, y_validering])
final_modell = LinearRegression()
final_modell.fit(X_final, y_final)

# Evaluer deretter med testsettet. Skal gjøres til slutt, kun en gang. 
test_prediksjoner = final_modell.predict(X_test)
print("\n--- Test (2025) ---")
print("MAE:", mean_absolute_error(y_test, test_prediksjoner))
print("R2:", r2_score(y_test, test_prediksjoner))
print("RMSE:", root_mean_squared_error(y_test, test_prediksjoner))

#sjekker hvor bra modellen er til å predikere høye pollennivåer.
grense = 70
faktisk_høy_test = (y_test > grense).to_numpy()
prediksjon_høy_test = test_prediksjoner > grense

treff = (faktisk_høy_test & prediksjon_høy_test).sum()
bommet_toppdag = (faktisk_høy_test & ~prediksjon_høy_test).sum()
falsk_alarm = (~faktisk_høy_test & prediksjon_høy_test).sum()
riktig_lav = (~faktisk_høy_test & ~prediksjon_høy_test).sum()

print(f"Dager over {grense} : {faktisk_høy_test.sum()}")
print(f"Treff: {treff}")
print(f"Falsk alarm: {falsk_alarm}") 
print(f"Bommet toppdag: {bommet_toppdag}")
print(f"Riktig lav: {riktig_lav}")

#Deler dagene inn i lav, moderat og høy
klasse_faktisk = y_test.apply(pollenklasse)
klasse_prediksjon = pd.Series(test_prediksjoner).apply(pollenklasse)
print("Andel dager med riktig kategori:", (klasse_faktisk.to_numpy() == klasse_prediksjon.to_numpy()).mean())

#Confusion matrix
matrise = np.array([[treff, bommet_toppdag], 
                    [falsk_alarm, riktig_lav]])

visning = ConfusionMatrixDisplay(confusion_matrix=matrise, display_labels=["Høy", "Ikke høy"])
visning.plot(cmap="Blues")
plt.title(f"Confusion matrix for bjørk  (grense: {grense})")
plt.xlabel("Modellens prediksjon")
plt.ylabel("Faktisk pollen")
plt.savefig("Confusion_matrix_bjørk.png")
plt.show()
               

#Graf
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(test["dato"], y_test, label="Faktisk pollen")
plt.plot(test["dato"], test_prediksjoner, label="Predikert pollen")
plt.xlabel("Dato")
plt.ylabel("Bjørkepollen")
plt.title("Pollenprediksjon: Bjørk")
plt.axhline(70, color='red', linestyle='--', label='Grense (70)')
plt.legend()
plt.tight_layout()
plt.savefig("Linear_regression_bjørk.png", dpi=150)
plt.show()

#Koefissientene
print("\n--- Koefissienter ---")
for navn, koeff in zip(features, final_modell.coef_):
    print(f"{navn}: {koeff: .3f}")
print(f"Konstantledd: {final_modell.intercept_: .3f}")

#Ganger koeffisientene med standardavviket slik at variablene kan sammenlignes
viktighet = final_modell.coef_ * X_final.std()
viktighet = viktighet.abs().sort_values()
print(viktighet)