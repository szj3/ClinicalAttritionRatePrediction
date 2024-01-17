# Import packages
import numpy as np
import matplotlib
import pd
import pandas as pd
from datetime import datetime
import us
import uszipcode
matplotlib.rcParams['figure.figsize'] = (10, 10)


# first step: vectorize the columns to convert strings into floats
# grab the columns

clin_addresses = pd.read_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/updated_clinical_attrition.csv")

# define a function that converts the dates into just the month and year
def parse_date(date_str):
    if pd.isna(date_str):
        return np.nan  # Return null
    try: # if it's in a month day year format, convert to month year
        return datetime.strptime(date_str, '%B %d, %Y').strftime('%B %Y')
    except ValueError:
        try:
            # already in the month year format, keep it/no change
            return datetime.strptime(date_str, '%B %Y').strftime('%B %Y')
        except ValueError:
            return np.nan  # Return null if both formats fail

# Apply the parse_date function the dates columns
clin_addresses['New Completion Date'] = clin_addresses['Completion Date'].apply(parse_date)
clin_addresses['New Start Date'] = clin_addresses['Start Date'].apply(parse_date)

# convert from strings to integers
clin_addresses['New Completion Date'] = pd.to_datetime(clin_addresses['New Completion Date'], errors='coerce', format='%B %Y')
clin_addresses['New Start Date'] = pd.to_datetime(clin_addresses['New Start Date'], errors='coerce', format='%B %Y')

#  difference in days
clin_addresses['length_of_trial'] = (clin_addresses['New Completion Date'] - clin_addresses['New Start Date']).dt.days

# Drop the original start_date and completion_date columns
# clin_addresses = clin_addresses.drop(columns=['Start Date', 'Completion Date'])

# Save the updated DataFrame to a new CSV file
clin_addresses.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/updated_clinical_attrition.csv", index=False)

# add the ruca codes
ruca_codes = pd.read_excel("2006 Complete Excel RUCA file 3.xls")
clin_addresses['Zipcode'] = clin_addresses['Zipcode'].astype(str)
ruca_codes['ZIPA'] = ruca_codes['ZIPA'].astype(str)
final_table = pd.merge(clin_addresses, ruca_codes, left_on='Zipcode', right_on='ZIPA', how='left')
drop_columns = ['ZIPN', 'ZIPA', 'STNAME','COMMFLAG']
final_table = final_table.drop(columns=drop_columns)
final_table.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table.csv", index=False)

# define a function for the zipcodes
## if the zipcodes do not have exactly 5 digits
## if the zipcodes contain a letter
## if the zipcodes contain a "dash" take the first part

def cleaned_zipcodes(final_table):
    search = uszipcode.SearchEngine() # search engine comes with package
    clean_zipcode = []
    for i, city in zip(final_table['Zipcode'], final_table['City']): # zip zipcodes and city together
        if "-" in str(i):
            clean_zipcode.append(str(i).split('-')[0])
        elif pd.isna(i):
            result = search.by_city(city)
            clean_zipcode.append(result[0].zipcode if result else None)
        else:
            clean_zipcode.append(i)
    return clean_zipcode
final_table['Cleaned_Zipcodes'] = cleaned_zipcodes(final_table)
final_table.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table.csv", index=False)

def cleaned_states(final_table):
    clean_state = []
    for state in final_table['State']:
        if state in [str(s) for s in us.states.STATES]:
            clean_state.append(state)
        else:
            clean_state.append(None)
    return clean_state
final_table['Cleaned States'] = cleaned_states(final_table)
final_table.dropna(subset=['Cleaned States'], inplace=True)
final_table.drop(columns='Cleaned States', inplace=True)
final_table.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table.csv", index=False)

def convert_minimum_age(age):
    age = str(age)
    if "Years" in age:
        return int(''.join(filter(str.isdigit, age)))
    elif "Months" in age:
        return float(''.join(filter(str.isdigit, age))) / 12
    else:
        return None
final_table['Minimum Age'] = final_table['Minimum Age'].apply(convert_minimum_age)
final_table.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table.csv", index=False)
