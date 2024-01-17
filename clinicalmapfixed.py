from pyzipcode import ZipCodeDatabase
import folium
import pandas as pd

clin_addresses = pd.read_excel("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/project_testCleanedUSZip.xlsx")

zcdb = ZipCodeDatabase()
center_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

for index in clin_addresses.index:
    zip_code = str(clin_addresses.loc[index, 'zip'])

    # Remove the extended part from the ZIP code if it exists
    if '-' in zip_code:
        zip_code = zip_code.split('-')[0]

    location_name = clin_addresses.loc[index, 'location_name']
    nct_id = clin_addresses.loc[index, 'nct_id']
    dropout_rate = clin_addresses.loc[index, 'dropout_percentage_all']
    phase = clin_addresses.loc[index, 'phase']
    study_type = clin_addresses.loc[index, 'study_type']

    try:
        location = zcdb[zip_code]
        lon, lat = location.longitude, location.latitude
        coordinates = [lat, lon]

        folium.Marker(
            location=coordinates,
            popup=f"{location_name}\nNCT ID: {nct_id}\nDropout Rate: {dropout_rate}\nPhase: {phase}\nStudy Type: {study_type}"
        ).add_to(center_map)
    except KeyError:
        print(f"Couldn't find location for ZIP code: {zip_code}")

# Save the map to an HTML file
center_map.save('map-clinical_trials.html')
