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
df['timestamp_pollen'] = pd.to_datetime(df['timestamp_pollen'])
df['timestamp_weather'] = pd.to_datetime(df['timestamp_weather'])

df = df.rename(columns={
                        "station_abbr_pollen" : "Pollenstasjon",  
                        "station_abbr_weather" : "Værstasjon",
                        "master_timestamp" : "Samlende_tid",
                        "timestamp_pollen": "Pollen_tid",
                        "timestamp_weather" : "Vær_tid",        
                        "kabetuh0" : "Bjørk", 
                        "khpoach0" : "Gress",
                        "sre000h0": "Sol", 
                        "erefaoh0" : "Nedbør",
                        "dkl010h0" : "Vind",
                        "tre200h0" : "LuftTemp"
                        })

df = df.drop(columns=["kaalnuh0","kacoryh0","kafaguh0",
                      "kafraxh0","kaquerh0","tre200hn",
                      "tre200hx","tre005h0","tre005hn",
                      "ure200h0","pva200h0","tde200h0",
                      "prestah0","pp0qffh0","pp0qnhh0",
                      "ppz700h0","ppz850h0","fkl010h1",
                      "fkl010h0","fu3010h0","fu3010h1",
                      "fkl010h3","fu3010h3","wcc006h0",
                      "fve010h0","rre150h0","htoauths",
                      "gre000h0","oli000h0","olo000h0",
                      "osr000h0","tso005hs","tso010hs",
                      "tso020hs", "ods000h0"])
print(df.head(10))
"print(df.describe())"
"print(df.dtypes)"

"Trening her ...."