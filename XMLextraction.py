import requests  # allows you to send HTTP/1.1 requests extremely easily.
from bs4 import BeautifulSoup
import pandas as pd


# Step 1: Read the CSV File of attrition rates with corresponding NCT Ids
clinical_attrition = pd.read_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/ct_attrition_dataset.csv")
clinical_attrition.head()

# Step 2: visualize the data by creating a boxplot
# datavisualize = clinical_attrition.boxplot(column="dropout_percentage_all", by="nct_id")
# print(datavisualize)

# Step 3 & 4: Create the URL for each clinical trial and retrieve its XML file
new_data = []  # store the new data in an empty list for now.
for index in clinical_attrition.index: # loop through each nct id and create a url from it
    nct_id = clinical_attrition.loc[index, 'nct_id']  # integer-location based indexing. accesses elements by their integer position.
    url= "https://clinicaltrials.gov/ct2/show/" + nct_id + "?resultsxml=true"  # create the url
    response = requests.get(url) # returns the nct urls
    if response.status_code == 200:  # 200 is the HTTP status code for "OK", a successful response.
        xmlcontent = BeautifulSoup(response.content, features="xml")

        # Step 4: Extract information from the XML file. Web scrapping

        # trial characteristics
        title_clinicaltrial = xmlcontent.find('brief_title').text if xmlcontent.find('brief_title').text else None
        # title of the trial
        overall_status = xmlcontent.find('overall_status').text if xmlcontent.find('overall_status').text else None
        # status of the trial
        phase_trial = xmlcontent.find('phase').text if xmlcontent.find('phase') else None
        # phase of trial
        allocation = xmlcontent.find('allocation').text if xmlcontent.find('allocation') else None
        # the type of study design
        start_date = xmlcontent.find('start_date').text if xmlcontent.find('start_date') else None
        # start date of trial
        completion_date = xmlcontent.find('completion_date').text if xmlcontent.find('completion_date') else None
        # completion date
        primary_purpose = xmlcontent.find('primary_purpose').text if xmlcontent.find('primary_purpose') else None
        # primary purpose of trial

        # location: city, state, zipcode
        city = xmlcontent.find('city').text if xmlcontent.find('city') is not None else None
        state = xmlcontent.find('state').text if xmlcontent.find('state') is not None else None
        zipcode = xmlcontent.find('zip').text if xmlcontent.find('zip') is not None else None
        facility = xmlcontent.find('facility').text if xmlcontent.find('facility') is not None else None

        # patient demographics: minimum age, maximum age, gender
        gender = xmlcontent.find('gender').text if xmlcontent.find('gender') else None
        minimum_age = xmlcontent.find('minimum_age').text if xmlcontent.find('minimum_age') else None
        maximum_age = xmlcontent.find('maximum_age').text if xmlcontent.find('maximum_age') else None

        # Step 5: Update the clinical_attrition dataframe with new information. Dictionary created
        new_data.append({
            'Clinical Title': title_clinicaltrial,
            'Overall Status': overall_status,
            'Trial Phase': phase_trial,
            'Allocation': allocation,
            'Start Date': start_date,
            'Completion Date': completion_date,
            'Primary Purpose': primary_purpose,
            'City': city,
            'State': state,
            'Zipcode': zipcode,
            'Gender': gender,
            'Minimum Age': minimum_age,
            'Maximum Age': maximum_age
        })
    else:
        print(f"No XML file for this clinical trial {nct_id}")
clinical_attrition = clinical_attrition.assign(**pd.DataFrame(new_data))
# Save the updated DataFrame to a new CSV file
clinical_attrition.to_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/updated_clinical_attrition.csv", index=False)
clinical_attrition.head()