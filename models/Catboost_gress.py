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

"----------GRESS------------"

avg_gress = df.groupby("dato")["Gress"].agg(gress_mean ='mean', gress_sum = 'sum', gress_count ='count')

avg_total_sun = df.groupby("dato")["Sol (minutt)"].agg(sol_mean ='mean', sol_sum = 'sum', sol_count ='count')

avg_temp = df.groupby("dato")["LuftTemp"].agg(temp_mean ='mean', temp_sum = 'sum', temp_count ='count')

avg_regn = df.groupby("dato")["Nedbør (mm)"].agg(regn_mean ='mean', regn_sum = 'sum', regn_count ='count')

avg_vind = df.groupby("dato")["Vind (m/s?)"].agg(vind_mean ='mean', vind_sum = 'sum', vind_count ='count')

avg_luftfuktighet = df.groupby("dato")["LuftFuktighet"].agg(luftfukt_mean ='mean', luftfukt_sum = 'sum', luftfukt_count ='count')

avg_dagnr = df.groupby("dato")["day_date"].first().rename("dagnr")

average_gress = pd.concat([avg_dagnr, avg_gress, avg_total_sun, avg_temp, avg_regn, avg_vind, avg_luftfuktighet], axis=1) 

avg_dagnr = df.groupby("dato")["day_date"].first().rename("dagnr")
avg_gress = avg_gress[avg_gress['gress_count']>=20]

pollen_igår = (
    avg_gress["gress_mean"]  # Bytt til kolonnen med selve pollenverdien hvis dette er en målingsteller
    .where(avg_gress["gress_count"] >= 20)
    .shift(1)
    .rename("pollen_igår")
)

total_sun = avg_total_sun[avg_total_sun["sol_sum"] >= 20] #fiks dette
avg_temp = avg_temp[avg_temp["temp_count"] >= 20]
avg_regn = avg_regn[avg_regn["regn_count"] >= 20]
avg_vind = avg_vind[avg_vind["vind_count"] >= 20]
avg_luftfuktighet = avg_luftfuktighet[avg_luftfuktighet["luftfukt_count"] >= 20]

average_gress = pd.concat([avg_dagnr, avg_gress, pollen_igår, total_sun, avg_temp, avg_regn, avg_vind, avg_luftfuktighet ], axis=1) 

"---------------End of table cleaning -----------------------"

import numpy as np
import pandas as pd
import catboost as cb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Rader ved starten:", len(average_gress))
print("Kolonner:", average_gress.columns.tolist())

# Tidsfeatures
average_gress["kalender_dag"] = pd.to_datetime(average_gress.index).day
average_gress["mnd"] = pd.to_datetime(average_gress.index).month
average_gress["år"] = pd.to_datetime(average_gress.index).year

features = ['temp_mean', 'luftfukt_mean', 'sol_sum', 'pollen_igår' ]
target = 'gress_mean'

print("Før dropna:", len(average_gress))
print(average_gress[features + [target]].isna().sum())

# Dropp NaN i features og target
average_gress.dropna(subset=features + [target], inplace=True)

# Del data kronologisk
train_early = average_gress[average_gress["år"] <= 2024]
val_2025 = average_gress[average_gress["år"] == 2025]
train_final = average_gress[average_gress["år"] <= 2025]
test_2026 = average_gress[average_gress["år"] == 2026]

# Kontroller at alle delene inneholder data
print("Trening til og med 2023:", len(train_early))
print("Validering 2024:", len(val_2025))
print("Endelig trening til og med 2024:", len(train_final))
print("Test 2025:", len(test_2026))

if train_early.empty or val_2025.empty or train_final.empty or test_2026.empty:
    raise ValueError("En av datasplittende er tom. Kontroller hvilke år som finnes i dataene.")


model_for_validation = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

model_for_validation.fit(
    train_early[features],
    train_early[target],
    eval_set=(val_2025[features], val_2025[target]),
    early_stopping_rounds=50,
    verbose=100
)


best_iterations = model_for_validation.get_best_iteration() + 1
print("Valgt antall iterasjoner:", best_iterations)


final_model = CatBoostRegressor(
    iterations=best_iterations,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

final_model.fit(
    train_final[features],
    train_final[target]
)



predictions_2026 = final_model.predict(test_2026[features])
predictions_2026 = np.maximum(predictions_2026, 0)

y_test = test_2026[target]

mae = mean_absolute_error(y_test, predictions_2026)
rmse = np.sqrt(mean_squared_error(y_test, predictions_2026))

print(f"MAE for 2025: {mae:.2f}")
print(f"RMSE for 2025: {rmse:.2f}")

"-----------------"

# Indeks = dato
test_dates = pd.to_datetime(test_2026.index)

# periode
periode = (test_dates >= "2026-04-01") & (test_dates <= "2026-09-09")


y_test_periode = y_test.iloc[periode]
predictions_periode = predictions_2026[periode]

mae_periode = mean_absolute_error(y_test_periode, predictions_periode)
rmse_periode = np.sqrt(mean_squared_error(y_test_periode, predictions_periode))

print(f"Antall dager i perioden: {periode.sum()}")
print(f"MAE (1. mai–1. sept 2026): {mae_periode:.2f}")
print(f"RMSE 1. mai–1. sept 2026): {rmse_periode:.2f}")

typisk_pollenverdi = y_test_periode.mean()

print(f"Gjennomsnittlig faktisk pollenverdi: {typisk_pollenverdi:.2f}")
print(f"MAE / gjennomsnitt: {mae_periode / typisk_pollenverdi * 100:.1f}%")
print(f"RMSE / gjennomsnitt: {rmse_periode / typisk_pollenverdi * 100:.1f}%")


print("\nFørste 10 prediksjoner for 2026:", predictions_2026[:10])

"------ Beregning av når pred og reele målinger er 90+ for 2025 -----------" 

# Lag én serie med prediksjoner indeksert etter dato
prediksjoner = pd.Series(
    predictions_2026,
    index=test_2026.index,
    name="predikert_pollen"
)

# Bygg tabellen med samme datoutvalg for alle kolonnene
sammenligningstabell = pd.DataFrame({
    "dato": test_dates[periode],
    "predikert_pollen": prediksjoner.iloc[periode].to_numpy(),
    "målt_pollen": y_test.iloc[periode].to_numpy(),
})

"""mae_høy_pollen = mean_absolute_error(sammenligningstabell["predikert_pollen_cat"], sammenligningstabell["målt_pollen_cat"])"""

høy_mask = sammenligningstabell["målt_pollen"] >= 20

if høy_mask.any():
    mae_høy_pollen = mean_absolute_error(sammenligningstabell.loc[høy_mask, "målt_pollen"],sammenligningstabell.loc[høy_mask, "predikert_pollen"])
    rmse_høy_pollen = np.sqrt(mean_squared_error(sammenligningstabell.loc[høy_mask, "målt_pollen"],sammenligningstabell.loc[høy_mask, "predikert_pollen"])
    )
    print(f"Antall dager i perioden: {periode.sum()}")
    print(f"Antall dager i rmse perioden: {høy_mask.sum()}")
    print(f"Høy pollen MAE: {mae_høy_pollen:.2f}")
    print(f"Høy pollen RMSE: {rmse_høy_pollen:.2f}")
else:
    print("Ingen dager med målt pollenverdi på minst 20 i perioden.")


faktisk_høy = sammenligningstabell["målt_pollen"] >= 20
predikert_høy = sammenligningstabell["predikert_pollen"] >= 20

treff = (faktisk_høy & predikert_høy).sum()
falsk_alarm = (~faktisk_høy & predikert_høy).sum()
bommet_toppdag = (faktisk_høy & ~predikert_høy).sum()
riktig_lav = (~faktisk_høy & ~predikert_høy).sum()

print(f"Faktisk og predikert >= 20 (treff): {treff}")
print(f"Predikert >= 20, men målt < 20 (falske alarmer): {falsk_alarm}")
print(f"Målt >= 20, men predikert < 20 (bommet toppdag): {bommet_toppdag}")
print(f"Både målt og predikert < 20: {riktig_lav}")

"---sjekk for overtilpassing , lekasjer osv.. ---"

"------------------"

# Visualisering
plt.figure(figsize=(14, 6))

plt.plot(
    train_final.index,
    train_final[target],
    label="Historisk gresspollen (til og med 2025)",
    color="blue",
    alpha=0.4,
)

plt.plot(
    test_2026.index,
    y_test,
    label="Faktisk gresspollen 2026",
    color="black",
    linewidth=1.5,
)

plt.plot(
    test_2026.index,
    predictions_2026,
    label="CatBoost-prediksjon 2026",
    color="green",
    linestyle="--",
    linewidth=2,
)

plt.title("Gresspollen 2026: faktisk vs. CatBoost-prediksjon")
plt.xlabel("Dato")
plt.ylabel("Pollenkonsentrasjon (gress_mean)")
plt.axvline(
    pd.Timestamp("2026-01-01"),
    color="red",
    linestyle=":",
    label="Starten på 2026",
)

plt.axhline(y=20, color='r', linestyle='-', label="Høyt nivå")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

"https://stackoverflow.com/questions/33382619/plot-a-horizontal-line-on-a-given-plot"