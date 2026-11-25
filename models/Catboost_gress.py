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

features = ['temp_mean', 'luftfukt_mean', 'sol_sum', 'pollen_igår', 'dagnr' ]
target = 'gress_mean'

verdier_til_dictionary = {}
verdier_til_dictionary["før_dropna"] = len(average_gress)
average_gress.dropna(subset=features + [target], inplace=True)
verdier_til_dictionary["etter_dropna"] = len(average_gress)

# Del data kronologisk
train_early = average_gress[average_gress["år"] <= 2024]
val_2025 = average_gress[average_gress["år"] == 2025]
train_final = average_gress[average_gress["år"] <= 2025]
test_2026 = average_gress[average_gress["år"] == 2026]


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

predictions_2025 = model_for_validation.predict(val_2025[features])
predictions_2025 = np.maximum(predictions_2025, 0)

y_val_2025 = val_2025[target]

val_dates_2025 = pd.to_datetime(val_2025.index)
periode_2025 = (
    (val_dates_2025 >= "2025-03-01") &
    (val_dates_2025 <= "2025-05-31")
)

predictions_2025 = predictions_2025[periode_2025]
y_val_2025 = y_val_2025.iloc[periode_2025]
mae_2025 = mean_absolute_error(y_val_2025, predictions_2025)
rmse_2025 = np.sqrt(mean_squared_error(y_val_2025, predictions_2025))
predictions_2026 = final_model.predict(test_2026[features])
predictions_2026 = np.maximum(predictions_2026, 0)
y_test = test_2026[target]

mae = mean_absolute_error(y_test, predictions_2026)
rmse = np.sqrt(mean_squared_error(y_test, predictions_2026))

"-----------------"

# Indeks = dato
test_dates = pd.to_datetime(test_2026.index)
# periode
periode = (test_dates >= "2026-04-01") & (test_dates <= "2026-09-09")
y_test_periode = y_test.iloc[periode]
predictions_periode = predictions_2026[periode]
mae_periode = mean_absolute_error(y_test_periode, predictions_periode)
rmse_periode = np.sqrt(mean_squared_error(y_test_periode, predictions_periode))

"------ Beregning av når pred og reele målinger for 2026 -----------" 

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
    "predikert_nivå" : None,
})

none = sammenligningstabell["målt_pollen"]  == 0
low = sammenligningstabell["målt_pollen"].between(1, 19, inclusive="both") 
medium = sammenligningstabell["målt_pollen"].between(20, 49, inclusive="both") 
høy = sammenligningstabell["målt_pollen"] >= 50

none_pred = sammenligningstabell["predikert_pollen"]  == 0
low_pred = sammenligningstabell["predikert_pollen"].between(1, 19, inclusive="both") 
medium_pred = sammenligningstabell["predikert_pollen"].between(20, 49, inclusive="both") 
høy_pred = sammenligningstabell["predikert_pollen"] >= 50

i = 0
while i < len(sammenligningstabell):
    if none.iloc[i]:
        sammenligningstabell.loc[i, "predikert_nivå"] = "NONE"
    elif low.iloc[i]:
        sammenligningstabell.loc[i, "predikert_nivå"] = "LOW"
    elif medium.iloc[i]:
        sammenligningstabell.loc[i, "predikert_nivå"] = "MEDIUM"
    elif høy.iloc[i]:
        sammenligningstabell.loc[i, "predikert_nivå"] = "HIGH"
    i += 1

#LAGER predikert_gress.csv for pollen-lookup.py
from pathlib import Path
csv_fil = Path(__file__).with_name("predikert_gress.csv")
sammenligningstabell["dato"] = pd.to_datetime(sammenligningstabell["dato"]).dt.strftime("%d.%m.%Y")
sammenligningstabell.to_csv(csv_fil, index=False)

#SANNHETSTABELLER
if none.any():
    mae_none_pollen = mean_absolute_error(sammenligningstabell.loc[none, "målt_pollen"], sammenligningstabell.loc[none, "predikert_pollen"])
    rmse_none_pollen = np.sqrt(mean_squared_error(sammenligningstabell.loc[none, "målt_pollen"],sammenligningstabell.loc[none, "predikert_pollen"])
    )

faktisk_none = sammenligningstabell["målt_pollen"] == 0
predikert_none = sammenligningstabell["predikert_pollen"] == 0
treff_null = (faktisk_none & predikert_none).sum()
falsk_alarm_null = (~faktisk_none & predikert_none).sum()
bommet_null = (faktisk_none & ~predikert_none).sum()
riktig_ikke_null = (~faktisk_none & ~predikert_none).sum()

    # Beregner lavt pollennivå:
if low.any():
    y_true = sammenligningstabell.loc[low, "målt_pollen"]
    y_pred = sammenligningstabell.loc[low, "predikert_pollen"]

    mae_low_pollen = mean_absolute_error(y_true, y_pred)
    rmse_low_pollen = np.sqrt(mean_squared_error(y_true, y_pred))

faktisk_low = sammenligningstabell["målt_pollen"].between(1, 19, inclusive="both")
predikert_low = sammenligningstabell["predikert_pollen"].between(1, 19, inclusive="both")
treff_low = (faktisk_low & predikert_low).sum()
falsk_alarm_low = (~faktisk_low & predikert_low).sum()
bommet_low = (faktisk_low & ~predikert_low).sum()
riktig_ikke_low = (~faktisk_low & ~predikert_low).sum()
    
    # Beregner medium pollennivå:
if medium.any():
    y_true = sammenligningstabell.loc[medium, "målt_pollen"]
    y_pred = sammenligningstabell.loc[medium, "predikert_pollen"]
    mae_medium_pollen = mean_absolute_error(y_true, y_pred)
    rmse_medium_pollen = np.sqrt(mean_squared_error(y_true, y_pred))

faktisk_medium = sammenligningstabell["målt_pollen"].between(20, 49, inclusive="both")
predikert_medium = sammenligningstabell["predikert_pollen"].between(20, 49, inclusive="both")
treff_medium = (faktisk_medium & predikert_medium).sum()
falsk_alarm_medium = (~faktisk_medium & predikert_medium).sum()
bommet_medium = (faktisk_medium & ~predikert_medium).sum()
riktig_ikke_medium = (~faktisk_medium & ~predikert_medium).sum()

    # Beregner høy pollennivå:
if høy.any():
    mae_høy_pollen = mean_absolute_error(sammenligningstabell.loc[høy, "målt_pollen"], sammenligningstabell.loc[høy, "predikert_pollen"])
    rmse_høy_pollen = np.sqrt(mean_squared_error(sammenligningstabell.loc[høy, "målt_pollen"],sammenligningstabell.loc[høy, "predikert_pollen"])
    )

faktisk_høy = sammenligningstabell["målt_pollen"] >= 50
predikert_høy = sammenligningstabell["predikert_pollen"] >= 50
treff_high = (faktisk_høy & predikert_høy).sum()
falsk_alarm_high = (~faktisk_høy & predikert_høy).sum()
bommet_toppdag = (faktisk_høy & ~predikert_høy).sum()
riktig_lav = (~faktisk_høy & ~predikert_høy).sum()

"------------------"

# ---------------Printout for å se verdier ----------
trening_val_test_dager_antall = {
    "Antall dager" : verdier_til_dictionary["før_dropna"], 
    "Antall dager uten nullverdier": verdier_til_dictionary["etter_dropna"],
    "Antall - trening 2023" : len(train_early), 
    "Antall - validering 2024" : len(val_2025),
    "Antall - endelig trening 2023-2024" : len(train_final),
    "Beste iterasjoner" : best_iterations,
    }

val_prediksjoner_2025 = {
    "Antall dager for val-prediksjon" : len(y_val_2025),
    "MAE for 2025" : round(float(mae_2025),2), 
    "RMSE for 2025": round(float(rmse_2025),2),
}
pollen_gress_prediksjoner_2026_periode = {
"Antall dager i prediksjonsperioden (1. april - 9.sept 2026)" : periode.sum(),
"MAE" : round(float(mae_periode),2),
"RMSE": round(float(rmse_periode),2),
}
    #""""Holdes separat === "Antall dager i perioden": round(float(periode.sum),2),"""""
pollenmengde_none = {
        "Antall dager i rmse perioden": round(float(none.sum()), 2),
        "Null pollen MAE": round(float(mae_none_pollen), 2),
        "Null pollen RMSE": round(float(rmse_none_pollen), 2),
        "Treff": round(float(treff_null), 2),
        "Falsk alarm": round(float(falsk_alarm_null), 2),
        "Bom": round(float(bommet_null), 2),
        "Riktig_ikke_null": round(float(riktig_ikke_null), 2)
    }

pollenmengde_low = {
        "Antall dager i rmse perioden": round(float(low.sum()), 2),
        "Lav pollen MAE": round(float(mae_low_pollen), 2),
        "Lav pollen RMSE": round(float(rmse_low_pollen), 2),
        "Treff": round(float(treff_low), 2),
        "Falsk alarm": round(float(falsk_alarm_low), 2),
        "Bom": round(float(bommet_low), 2),
        "Riktig_ikke_null": round(float(riktig_ikke_low), 2)
    }

pollenmengde_medium = {
        "Antall dager i rmse perioden": round(float(medium.sum()), 2),
        "Medium pollen MAE": round(float(mae_medium_pollen), 2),
        "Medium pollen RMSE": round(float(rmse_medium_pollen), 2),
        "Treff": round(float(treff_medium), 2),
        "Falsk alarm": round(float(falsk_alarm_medium), 2),
        "Bom": round(float(bommet_medium), 2),
        "Riktig_ikke_null": round(float(riktig_ikke_medium), 2)
    }

pollenmengde_high = {
        "Antall dager i rmse perioden": round(float(høy.sum()), 2),
        "Høy pollen MAE": round(float(mae_høy_pollen), 2),
        "Høy pollen RMSE": round(float(rmse_høy_pollen), 2),
        "Treff": round(float(treff_high), 2),
        "Falsk alarm": round(float(falsk_alarm_high), 2),
        "Bom": round(float(bommet_toppdag), 2),
        "Riktig_ikke_null": round(float(riktig_lav), 2) #print(f"Både målt og predikert < 70: {riktig_lav}")"""
    }

def print_antall_dager_og_målinger():
    print(trening_val_test_dager_antall)
    print(val_prediksjoner_2025)
    print(pollen_gress_prediksjoner_2026_periode)

def print_pollenmengder():
    print(pollenmengde_none)
    print(pollenmengde_low)
    print(pollenmengde_medium)
    print(pollenmengde_high)

print_antall_dager_og_målinger()
print_pollenmengder()

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

plt.axhline(y=20, color='orange', linestyle='-', label="Medium nivå")
plt.axhline(y=50, color='red', linestyle='-', label="Høyt nivå")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

"https://stackoverflow.com/questions/33382619/plot-a-horizontal-line-on-a-given-plot"