import pandas as pd
import numpy as np


def cleanup(file: str) -> pd.DataFrame:
    df = pd.read_csv('time_series_covid19_deaths_US.csv')
    unique_states = np.unique(df["Province_State"].values)
    states = np.array([
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
        'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
        'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
        'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
        'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
    ])
    not_states = set(unique_states).difference(set(states))
    df = df[~df["Province_State"].isin(not_states)]
    df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
                  'Country_Region', 'Lat', 'Long_', 'Combined_Key'], axis=1)
    df = df.groupby("Province_State").agg(sum)
    df.reset_index(inplace=True)
    return df
