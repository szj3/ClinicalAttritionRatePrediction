# install all the packages
import requests
from bs4 import BeautifulSoup
from pyzipcode import ZipCodeDatabase
import folium
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

clin_addresses = pd.read_excel("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table.csv")
ruca_codes = pd.read_excel("2006 Complete Excel RUCA file 3.xls")

valid_zip_codes = []
for index in ruca_codes.index:
    ruca_value = ruca_codes.loc[index, 'RUCA2.0']  # accessing RUCA2.0
    if ruca_value in [7.0, 7.2, 7.3, 7.4, 8.0, 8.2, 8.3, 8.4, 9.0, 9.1, 9.2, 10.0, 10.2, 10.3, 10.4, 10.5, 10.6]:
        valid_zip_codes.append(ruca_codes.loc[index, 'ZIPA'])
valid_zip_codes = set(valid_zip_codes)

zcdb = ZipCodeDatabase()
center_map = folium.Map(location=[39.8283, -98.5795], zoom_start=8)  # google the center of the US

for index in clin_addresses.index:
    zip_code = str(clin_addresses.loc[index, 'zip'])
    if '-' in zip_code:
        zipcode_formatted = zip_code.split('-')[0] # splitting the zipcode if in the #####-####
        print(zipcode_formatted)
    else:
        print("Unable to provide location for this clinical trial")
        continue # skip to the next zipcode

    location_name = clin_addresses.loc[index, 'location_name'] # grab the info I want
    nct_id = clin_addresses.loc[index, 'nct_id']
    dropout_rate = clin_addresses.loc[index, 'dropout_percentage_all']
    phase = clin_addresses.loc[index, 'phase']
    study_type = clin_addresses.loc[index, 'study_type']

    try:
        location = zcdb[zipcode_formatted]
        lon = location.longitude
        lat = location.latitude
        coordinates = [lat, lon]
        folium.Marker(
            location=coordinates, # grab items for the pop-up
            popup=f"{location_name}\nNCT ID: {nct_id}\nDropout Rate: {dropout_rate}\nPhase: {phase}\nStudy Type: {study_type}"
        ).add_to(center_map)
    except KeyError:
        print(f"Couldn't find zipcode: '{zipcode_formatted}'")

center_map.save('map-clinical_trials.html')
