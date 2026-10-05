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
"print(df.head(10))"
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
print(average_bjørk.sample(10))
"print(average_gress.sample(10))"

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
end_date_train = pd.Timestamp('2024-12-31')

mask = average_bjørk['dato'].between(start_date_train, end_date_train)
trening_2023_24 = average_bjørk.loc[mask]
"print(trening_2023_24)"

start_date_val = pd.Timestamp('2025-01-01')
end_date_val = pd.Timestamp('2025-12-31')

mask = average_bjørk['dato'].between(start_date_val, end_date_val)
validering_2025 = average_bjørk.loc[mask]
"print(validering_2025)"

start_date_pred = pd.Timestamp('2026-01-01')
end_date_pred = pd.Timestamp('2026-09-09')
mask = average_bjørk['dato'].between(start_date_val, end_date_val)
actual_vals_2026 = average_bjørk.loc[mask]

"------ Manglende målinger. F.eks mellom den 15.-27.april er det null målinger ------"

"""mask_all_nan = avg_bjork['bjørk_mean'].isna()
if mask_all_nan.any():
    print("Grupper der alle 'Bjørk'-verdier er NaN:")
    print(avg_bjork[mask_all_nan])"""


"---------------End of table cleaning -----------------------"

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"Trening her ...."
"print(average_bjørk.columns.tolist())"


"--- Kilder for senere referanse ---"
"https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas"
"https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe"
"https://medium.com/@prathik.codes/mastering-randomforestregressor-in-scikit-learn-a-practical-guide-ba8615097100"


"https://dev.to/tamalbarman/running-a-random-forest-using-python-1920"
"https://weatherstory.io/pollen/guide/pollen-count-scale"

"https://www.youtube.com/watch?v=5_dnYnDCRXQ"
"https://www.youtube.com/watch?v=-aCF0_wfVwY"
