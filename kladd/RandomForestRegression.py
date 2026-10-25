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
df['timestamp_pollen'] = pd.to_datetime(df['timestamp_pollen'])
df['timestamp_weather'] = pd.to_datetime(df['timestamp_weather'])
df["dato_p"] = df["timestamp_pollen"].dt.date
df["dato_w"] = df["timestamp_weather"].dt.date
df["indeks"] = pd.DataFrame.iloc


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
print(df.head(10))
"print(df.describe())"
"print(df.dtypes)"


"--------BJØRK----------"

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



"----------GRESS------------"

avg_gress = df.groupby("dato")["Gress"].agg(gress_mean ='mean', gress_sum = 'sum', gress_count ='count')
if (avg_gress['gress_count'] >= 20).any():
    df.merge(avg_gress, left_on='dato', right_index=True, how='left')

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

average_bjørk = pd.concat([avg_bjork, avg_gress, avg_sun, avg_temp, avg_regn, avg_vind, avg_luftfuktighet], axis=1) 
average_gress = pd.concat([avg_gress, avg_sun, avg_temp, avg_regn, avg_vind, avg_luftfuktighet], axis=1) 
"print(average_bjørk.tail)"
print(average_gress.sample(10))

print("---Kun høye verdier ----")
høy_bjørk = average_bjørk["bjørk_mean"] >= 90
if(høy_bjørk).any():
 "print(average_bjørk[høy_bjørk])"

høy_gress = average_gress["gress_mean"] >= 20
if(høy_gress).any():
 "print(average_gress[høy_gress])"

"----Data split----"
average_bjørk = average_bjørk.reset_index()

average_bjørk['dato'] = pd.to_datetime(average_bjørk['dato'])

start_date_train = pd.Timestamp('2023-01-01')
end_date_train = pd.Timestamp('2025-12-31')

mask = average_bjørk['dato'].between(start_date_train, end_date_train)
trening_2023_25 = average_bjørk.loc[mask]
"print(trening_2023_24)"

"""start_date_val = pd.Timestamp('2025-01-01')
end_date_val = pd.Timestamp('2025-12-31')

mask = average_bjørk['dato'].between(start_date_val, end_date_val)
validering_2025 = average_bjørk.loc[mask]"""
"print(validering_2025)"

"""start_date_pred = pd.Timestamp('2026-01-01')
end_date_pred = pd.Timestamp('2026-09-09')
mask = average_bjørk['dato'].between(start_date_val, end_date_val)
actual_vals_2026 = average_bjørk.loc[mask]"""

"------ Manglende målinger. F.eks mellom den 15.-27.april er det null målinger ------"

"""mask_all_nan = avg_bjork['bjørk_mean'].isna()
if mask_all_nan.any():
    print("Grupper der alle 'Bjørk'-verdier er NaN:")
    print(avg_bjork[mask_all_nan])"""


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
average_bjørk['ukedag'] = average_bjørk["dato"].dt.day_of_week
average_bjørk['mnd'] = average_bjørk["dato"].dt.month
average_bjørk['år'] = average_bjørk["dato"].dt.year

# Inkluder tids-features i modellen din
features = ['sol_mean', 'temp_mean', 'regn_mean', 'vind_mean', 'luftfukt_mean', 'ukedag', 'mnd', 'år']
target = 'bjørk_mean'

print("Før dropna:", len(average_bjørk))
print(average_bjørk[features + [target]].isna().sum())

# Drop rows med manglende verdier i features eller target
average_bjørk.dropna(subset=features + [target], inplace=True)
"""print("Etter dropna:", len(average_bjørk))"""

# 2. KRONOLOGISK SPLITT (Korrekt oppsett)
# Treningsdata: Alt til og med 2025
train_df = average_bjørk[average_bjørk["år"] <= 2025]
X_train = train_df[features]
y_train = train_df[target]

# Testdata: De første 9 månedene av 2026
test_df = average_bjørk[average_bjørk["år"] == 2026]
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
predictions_2026 = model.predict(X_test)

# 6. EVALUERING
mae = mean_absolute_error(y_test, predictions_2026)
rmse = np.sqrt(mean_squared_error(y_test, predictions_2026))

"""print(f'\nMean Absolute Error (MAE) for 2026: {mae:.2f}')
print(f'Root Mean Squared Error (RMSE) for 2026: {rmse:.2f}')
"""
# 7. PRINT OG PLOT PREDIKSJONENE
print("\nFørste 10 prediksjoner for 2026:", predictions_2026[:10])

# Visualisering
plt.figure(figsize=(14, 6))
# Plot historisk trening (valgfritt, kan droppes hvis det blir for uoversiktlig)
plt.plot(train_df["dato"], y_train, label="Historisk Bjørk (2023-2025)", color="blue", alpha=0.4)
# Plot faktisk vs predikert for 2026
plt.plot(test_df["dato"], y_test, label="Faktisk Bjørk 2026", color="black", linewidth=1.5)
plt.plot(test_df["dato"], predictions_2026, label="CatBoost Prediksjon 2026", color="green", linestyle="--", linewidth=2)

plt.title("Bjørkepollen 2026: Faktisk vs CatBoost-prediksjon", fontsize=14, fontweight="bold")
plt.xlabel("Dato")
plt.ylabel("Pollenkonsentrasjon (bjørk_mean)")
plt.axvline(x=pd.to_datetime("2025-12-31"), color="red", linestyle=":", label="Starten på 2026")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

"--- Kilder for senere referanse ---"
"https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas"
"https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe"
"https://medium.com/@prathik.codes/mastering-randomforestregressor-in-scikit-learn-a-practical-guide-ba8615097100"


"https://dev.to/tamalbarman/running-a-random-forest-using-python-1920"
"https://weatherstory.io/pollen/guide/pollen-count-scale"

"https://www.youtube.com/watch?v=5_dnYnDCRXQ"
"https://www.youtube.com/watch?v=-aCF0_wfVwY"

"----META Prophet----"
"https://medium.com/@nia036/machine-learning-forecasting-with-meta-prophet-f7b3e1e50017"

"--Catboost--"
"https://www.geeksforgeeks.org/machine-learning/demand-forecasting-in-retail-using-catboost-1/"
"https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd/"
