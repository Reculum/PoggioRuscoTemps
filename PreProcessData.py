import pandas as pd
import numpy as np
from datetime import date

type = "train"

#trainset: 2014 - 2021
#testset: 2022 - 2024


df_dati = pd.DataFrame()

for anno in range(2014, 2022):

    df_dati = pd.concat([df_dati, pd.read_csv(f"./data/{anno}.csv", header=0)])


df_dati_rusco = df_dati[(df_dati["latitude"] == 44) & (df_dati["longitude"] == 11)].copy()


df_dati_rusco["time"] = pd.to_datetime(df_dati_rusco["time"]).dt.strftime("%d/%m/%y")
df_dati_rusco["t2m"] = df_dati_rusco["t2m"] - 273


df_dati_rusco = pd.DataFrame( {"date": (df_dati_rusco["time"]), "t2m":df_dati_rusco["t2m"]})
df_dati_rusco.to_csv(f"./data/{type}_data_rusco.csv", index=False)


def ComputeContinuosDate(day, month, year):

	FirstJan = date(year, 1, 1)
	Today = date(year, month, day)

	absoluteDayOfYear = (Today - FirstJan).days
	p = absoluteDayOfYear / 365

	return (np.cos(2 * p * np.pi), np.sin(2 * p * np.pi))




df_dati = pd.read_csv(f"./data/{type}_data_rusco.csv")
df_refined_rusco_data = pd.DataFrame({"xDate":[], "yDate":[], "t2m_now":[], "t2m_next":[]})


l = len(df_dati["t2m"])


for indx, x in df_dati.iterrows():

    if (indx != l - 1):

        [day, month, year] = x["date"].split("/", 2)
        

        xDate, yDate = ComputeContinuosDate( int(day), int(month), int(year))

        tmp_df = pd.DataFrame({"xDate":[xDate], 
                               "yDate":[yDate],  
                               "t2m_now":[x["t2m"]], 
                               "t2m_next":[df_dati["t2m"][indx + 1]]})

        
        df_refined_rusco_data = pd.concat([df_refined_rusco_data, tmp_df])

        # df_refined_rusco_data["day"]._append(giorno)
        # df_refined_rusco_data["month"].append(mese)
        # df_refined_rusco_data["year"].append(anno)

        # df_refined_rusco_data["t2m_now"].append(x["t2m"])
        # df_refined_rusco_data["t2m_next"].append(df_dati["t2m"][indx + 1])



df_refined_rusco_data.to_csv(f"./data/{type}_rusco_data.csv", index=False)