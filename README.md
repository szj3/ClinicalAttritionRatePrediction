This project was part of a final group project for class. The code written by me and the report was written in collaboration with my team members. 

Our objective was to predict the attrition percentage among patient enrollments in clinical trials and map the clinical research facilities geographically using information derived from ClinicalTrials.gov and its API.

We were given 1325 clinical trials that were identified by their NCT (National Clinical Trial number) numbers along with their corresponding attrition percentage. We needed to retrieve associated data from ClinicalTrials.gov API each clinical trial using the NCT number from our dataset.

To extract all 1325 clinical trials with NCT numbers, we defined and created a loop tasked to take all the 1325 NCT numbers and loop them each into the url space where the NCT number was written. This function fetches XML data for a specific clinical trial from ClinicalTrials.gov and it returns the XML data if the request is successful or nothing if the request failed. 

With XML data, we were able to examine the variables and features that were recorded and assessed from each clinical trial such as the study design, location, facility name, drug, eligibility criteria etc. This was important to select what features contributed to predicting the attrition percentage and what needed to be added to the existing dataset for more detailed information.

We added the gender of the participants, their minimum and maximum age, the start date of the trials, set completion date, trial study type, facility name, address, city, state, zip code, country, phase of the trial, status of the trials, and types of masking during the trial by creating a loop to extract the data of the XML data. This new data is now integrated into the dataset.

We merged (using lambda) the RUCA numerical code to the original dataset to categorize locations from urban to non-urban areas. We imported the RUCA files and had a table of the RUCA code associated with the zipcodes and changed the values to labels that state urban or non-urban. Using the NumPy library to create the new column to the dataframe, we added a column called ‘Dropout’ to help in manipulating the data and casting the prediction. This creates a binary classification outcome column based on the condition that if the dropout percentage column is greater than 7.66019, the corresponding value in the 'Dropout' column is set to 1; otherwise, it is set to 0. This helped predict the dropout percentages into whether the dropout percentage is above a certain threshold.

Having all the selected features, we converted the zipcodes (using the pyzipcode package) into their corresponding longitude and latitude codes to help with mapping the locations of where the trials took place. We then geocoded and looped all the facilities of each clinical trial using the longitude and latitude coordinates information into a HTML file (using folium package and a marker). The map showcased the map of the United States covered with drop pins of all the locations of the facilities. You can zoom in the map for accuracy. When you hover over the pin, it will show a text box with more information about the facility from the dataset. Because of high memory usage, we decided to visualize clinical trials that were above the median of 7.66019 rather than having all 818 trials on the map. 

When it comes to training our data, we prepare all the libraries to be used. We removed all the null values of the dataset to proceed further with the training and the column we don’t need in ML. We transformed the string data into the numbers for the machine to understand it. 

Files 
1. "XMLextraction.py": I extracted all the crucial information from all 1300 XML files into onto table, saved as a CSV file. 
2. "CleanData.py": This file was designated to clean the table by removing null values, standardizing the data, and filtering the trials that were only conducted in the United States.
3. "FoliumMapping.py": After cleaning the data, I mapped out all the facilities that were labeled as rural.
4. "PredictionModel.py": Seeks to predict the attrition rate of patients based on specific features from the table. 
