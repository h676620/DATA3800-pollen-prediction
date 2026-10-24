from datetime import date

import numpy as np
import pandas as pd
import csv
import DateTime
from sklearn.ensemble import RandomForestRegressor
"full_pollen_and_weather_data_Zurich.csv"
"Bjørk = kabetuh0 (høy = 90) ""Gress = khpoach0 (høy = 20) "
"Solskinn = sre000h0, "
"Regn = erefaoh0 , "
"Vind = dkl010h0, "
"Luft-temperatur = tre200h0"

df = pd.read_csv(('data/full_pollen_and_weather_data_Zurich.csv'), delimiter=";", quotechar='|')

"Konverter fra str til DateTime"
df['master_timestamp'] = pd.to_datetime(df['master_timestamp'])

df["dato"] = df["master_timestamp"].dt.date
df['day_date'] = df['master_timestamp'].dt.dayofyear
df['mnd'] = df['master_timestamp'].dt.month
df['yr'] = df['master_timestamp'].dt.year
df['timestamp_pollen'] = pd.to_datetime(df['timestamp_pollen'])
df['timestamp_weather'] = pd.to_datetime(df['timestamp_weather'])


"""https://dev.to/nilotpalc/answer-how-to-add-incremental-numbers-to-a-new-column-using-pandas-4bc5"""

df = df.rename(columns={
                        "station_abbr_pollen" : "Pollenstasjon",  
                        "station_abbr_weather" : "Værstasjon",
                        "master_timestamp" : "Samlende_tid",
                        "timestamp_pollen": "Pollen_tid",
                        "timestamp_weather" : "Vær_tid",        
                        "kabetuh0" : "Bjørk", 
                        "khpoach0" : "Gress",
                        "sre000h0": "Sol (minutt)", 
                        "rre150h0" : "Nedbør (mm)",
                        "fkl010h0" : "Vind (m/s?)",
                        "tre200h0" : "LuftTemp",
                        "ure200h0" : "LuftFuktighet"
                        })

df = df.drop(columns=["kaalnuh0","kacoryh0","kafaguh0",
                      "kafraxh0","kaquerh0","tre200hn",
                      "tre200hx","tre005h0","tre005hn",
                      "pva200h0","tde200h0","erefaoh0",
                      "prestah0","pp0qffh0","pp0qnhh0",
                      "ppz700h0","ppz850h0","fkl010h1",
                      "fu3010h0","fu3010h1",
                      "fu3010h3","wcc006h0",
                      "fve010h0","htoauths",
                      "gre000h0","oli000h0","olo000h0",
                      "osr000h0","tso005hs","tso010hs",
                      "tso020hs", "ods000h0", "Værstasjon", "Pollenstasjon"])
print(df.head(40))

avg_bjork = df.groupby("dato")["Bjørk"].agg(bjørk_mean ='mean', bjørk_sum = 'sum', bjørk_count ='count')
if (avg_bjork['bjørk_count'] >= 20 ).any():
    df.merge(avg_bjork, left_on='dato', right_index=True, how='left')

avg_sun = df.groupby("dato")["Sol (minutt)"].agg(sol_mean ='mean', sol_sum = 'sum', sol_count ='count')
if (avg_sun['sol_count'] >= 20).any():
    df.merge(avg_sun, left_on='dato', right_index=True, how='left')

avg_temp = df.groupby("dato")["LuftTemp"].agg(temp_mean ='mean', temp_sum = 'sum', temp_count ='count')
if (avg_temp['temp_count'] >= 20).any():
    df.merge(avg_temp, left_on='dato', right_index=True, how='left')

avg_regn = df.groupby("dato")["Nedbør (mm)"].agg(regn_mean ='mean', regn_sum = 'sum', regn_count ='count')
if (avg_regn['regn_count'] >= 20).any():
    df.merge(avg_regn, left_on='dato', right_index=True, how='left')

avg_vind = df.groupby("dato")["Vind (m/s?)"].agg(vind_mean ='mean', vind_sum = 'sum', vind_count ='count')
if (avg_vind['vind_count'] >= 20).any():
    df.merge(avg_vind, left_on='dato', right_index=True, how='left')

avg_luftfuktighet = df.groupby("dato")["LuftFuktighet"].agg(luftfukt_mean ='mean', luftfukt_sum = 'sum', luftfukt_count ='count')
if (avg_luftfuktighet['luftfukt_count'] >= 20).any():
    df.merge(avg_vind, left_on='dato', right_index=True, how='left')

avg_dagnr = df.groupby("dato")["day_date"].first().rename("dagnr")

average_bjørk = pd.concat([avg_dagnr, avg_bjork, avg_sun, avg_temp, avg_regn, avg_vind, avg_luftfuktighet ], axis=1) 
print(average_bjørk.sample(10))
"print(average_bjørk[average_bjørk.isna().any(axis=1)])"

"---------------End of table cleaning -----------------------"

import numpy as np
import pandas as pd
import catboost as cb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Rader ved starten:", len(average_bjørk))
print("Kolonner:", average_bjørk.columns.tolist())

# 1. Lag tids-features (Disse er superviktige for sesongbasert pollen!)
# Lag tids-features fra dato-indeksen
average_bjørk["kalender_dag"] = pd.to_datetime(average_bjørk.index).day
average_bjørk["mnd"] = pd.to_datetime(average_bjørk.index).month
average_bjørk["år"] = pd.to_datetime(average_bjørk.index).year

# Inkluder tids-features i modellen din
features = ['sol_mean', 'temp_mean', 'regn_mean', 'vind_mean', 'luftfukt_mean', 'dagnr']
target = 'bjørk_mean'

print("Før dropna:", len(average_bjørk))
print(average_bjørk[features + [target]].isna().sum())

# Drop rows med manglende verdier i features eller target
average_bjørk.dropna(subset=features + [target], inplace=True)
"""print("Etter dropna:", len(average_bjørk))"""

# 2. KRONOLOGISK SPLITT (Korrekt oppsett)
# Treningsdata: Alt til og med 2025
train_df = average_bjørk[average_bjørk["år"] <= 2024]
X_train = train_df[features]
y_train = train_df[target]

# Testdata: De første 9 månedene av 2026
test_df = average_bjørk[average_bjørk["år"] == 2025]
X_test = test_df[features]
y_test = test_df[target]

# Sjekk at vi har data i begge sett
"""print(f"Rader til trening (2023-2025): {len(X_train)}")
print(f"Rader til test (2026): {len(X_test)}")"""

# 3. INITIALISER CATBOOST REGRESSOR
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05, # Satt ned litt for mer stabil læring på tidsserier
    depth=6,
    loss_function='RMSE',
    eval_metric='MAE',
    random_seed=42,
    verbose=100
)

# 4. TREN MODELLEN
# Vi bruker 2026 som eval_set for å se hvordan den spår fremtiden underveis
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# 5. GENERER PREDIKSJONER FOR 2026
predictions_2025 = model.predict(X_test)
predictions_2025 = np.maximum(predictions_2025, 0)

# 6. EVALUERING
mae = mean_absolute_error(y_test, predictions_2025)
rmse = np.sqrt(mean_squared_error(y_test, predictions_2025))

print(f'\nMean Absolute Error (MAE) for 2026: {mae:.2f}')
print(f'Root Mean Squared Error (RMSE) for 2026: {rmse:.2f}')

# Sørg for at indeksene er datoer
test_dates = pd.to_datetime(test_df.index)

# Velg perioden 1. februar–1. mai 2025, inkludert begge datoene
periode = (test_dates >= "2025-03-01") & (test_dates <= "2025-05-10")

y_test_periode = y_test.iloc[periode]
predictions_periode = predictions_2025[periode]

mae_periode = mean_absolute_error(y_test_periode, predictions_periode)
rmse_periode = np.sqrt(mean_squared_error(y_test_periode, predictions_periode))

print(f"Antall dager i perioden: {periode.sum()}")
print(f"MAE (1. mars–10. mai 2025): {mae_periode:.2f}")
print(f"RMSE (1. mars–10. mai 2025): {rmse_periode:.2f}")

typisk_pollenverdi = y_test_periode.mean()

print(f"Gjennomsnittlig faktisk pollenverdi: {typisk_pollenverdi:.2f}")
print(f"MAE / gjennomsnitt: {mae_periode / typisk_pollenverdi * 100:.1f}%")
print(f"RMSE / gjennomsnitt: {rmse_periode / typisk_pollenverdi * 100:.1f}%")

# 7. PRINT OG PLOT PREDIKSJONENE
print("\nFørste 10 prediksjoner for 2025:", predictions_2025[:10])

# Visualisering
plt.figure(figsize=(14, 6))

plt.plot(
    train_df.index,
    y_train,
    label="Historisk Bjørk (til og med 2024)",
    color="blue",
    alpha=0.4,
)

plt.plot(
    test_df.index,
    y_test,
    label="Faktisk Bjørk 2025",
    color="black",
    linewidth=1.5,
)

plt.plot(
    test_df.index,
    predictions_2025,
    label="CatBoost-prediksjon 2025",
    color="green",
    linestyle="--",
    linewidth=2,
)

plt.title("Bjørkepollen 2025: Faktisk vs. CatBoost-prediksjon")
plt.xlabel("Dato")
plt.ylabel("Pollenkonsentrasjon (bjørk_mean)")
plt.axvline(pd.Timestamp("2025-01-01"), color="red", linestyle=":", label="Starten på 2025")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()