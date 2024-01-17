# Import packages
from collections import Counter
import numpy as np
import collections, numpy
import mlxtend
import matplotlib
import re
import pd
import pandas as pd
from datetime import datetime
import us
from matplotlib import pyplot
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10, 10)

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

final_table1 = pd.read_csv("/Users/shania/PycharmProjects/ClinicalAttritionRateMap/final_table1.csv")
# final_table1['dropout_percentage_all'].median() # = 7.6602

'''
final_table.isnull().sum()
# allocation has 116 NA, completion date has 31, primary purpose has 5, minimum is 9, maximum is 536, ruca is 219
final_table.drop(columns='Maximum Age', inplace=True)
final_table.drop(columns='Allocation', inplace=True)
final_table.drop(columns='Zipcode', inplace=True)
final_table.drop(columns='New Completion Date', inplace=True)
final_table.drop(columns='Completion Date', inplace=True)
final_table['Primary Purpose'].fillna(final_table['Primary Purpose'].mode()[0], inplace=True)
final_table['length_of_trial'].fillna(final_table['length_of_trial'].median(), inplace=True)
final_table['Minimum Age'].fillna(final_table['Minimum Age'].median(), inplace=True)
final_table['RUCA2.0'].fillna(final_table['RUCA2.0'].median(), inplace=True)
'''
final_table1['Dropout'] = np.where((final_table1['dropout_percentage_all'].apply(lambda x: float(x)))>7.660191019, 1, 0)
categorical_columns = final_table1[['Allocation', 'Trial Phase', 'Overall Status', 'Primary Purpose', 'City', 'State',
'Gender']]
# numerical column names and then categorical

# Convert the categorical data into numerical data
le = preprocessing.LabelEncoder()
X_2 = categorical_columns.apply(le.fit_transform)

# Apply OneHotEncoder
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
onehotlabels = enc.transform(X_2).toarray()

# Create a DataFrame with the one-hot encoded columns
onehot_df = pd.DataFrame(onehotlabels, columns=enc.get_feature_names_out(categorical_columns.columns))

# Concatenate the one-hot encoded DataFrame with the original DataFrame 'final_table1'
final_table1_encoded = pd.concat([final_table1, onehot_df], axis=1)
final_table1_encoded = final_table1_encoded.drop(categorical_columns.columns, axis=1)

# scale the columns using StandardScaler. this is only for numerical variables
numeric_columns = ['Cleaned_Zipcodes', 'RUCA2.0', 'length_of_trial', 'Minimum Age', 'New Start Date']

for column in numeric_columns:
    final_table1_encoded[column] = pd.to_numeric(final_table1_encoded[column], errors='coerce')
numeric_data = final_table1_encoded[numeric_columns]

# Scale the numeric columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
final_table1_encoded[numeric_columns] = scaled_data

excess_columns = final_table1_encoded[['nct_id', 'dropout_percentage_all', 'Clinical Title', 'Start Date', 'Completion Date',
                               'Zipcode', 'New Start Date', 'New Completion Date']]

final_table1_encoded = final_table1_encoded.drop(excess_columns.columns, axis=1) # table has been finalized.


# split the data and choose the column you want to train vs test
X_df = final_table1_encoded.drop('Dropout',axis=1)
y_df = final_table1_encoded['Dropout']
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=2)
Counter(y_train), Counter(y_test)

# logistical regression ML
#Fit the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
#Generate Predictions
predictions = model.predict(X_test)
predictions_proba = model.predict_proba(X_test)

plt.hist(predictions_proba[:,1])
# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(conf_mat=cm, show_absolute=True)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
#Accuracy
accuracy = (tp+tn)/(tp+tn+fp+fn)
print('Accuracy: %.3f' % accuracy)

#Recall/Sensitivity/True Positive rate
recall = sensitivity = tpr = tp / (tp + fn)
print('Recall: %.3f' % recall)

#Precision
precision = tp / (tp + fp)
print('Precision: %.3f' % precision)

#Specificity/Negative Recall/ True negative Rate/ 1-False Positive Rate
specificity = tn / (tn + fp)
print('Specificity: %.3f' % specificity)

#F1 Score
f1 = 2*(precision*recall)/(precision+recall)
print('F1: %.3f' % f1)

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.3f' % accuracy)

#Recall
recall = recall_score(y_test, predictions)
print('Recall: %.3f' % recall)

#Precision
precision = precision_score(y_test, predictions)
print('Precision: %.3f' % precision)

#The f1-score is the harmonic mean of precision and recall
f1 = f1_score(y_test, predictions)
print('F1: %.3f' % f1)

#AUROC = Area Under the Receiver Operating Characteristic curve
roc_auc = roc_auc_score(y_test, predictions_proba[:,1])
print('AUCROC: %.3f' % roc_auc)

print(classification_report(y_test, predictions))
# Random Forest
#Fit the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

#Prediction
predictions_proba = model.predict_proba(X_test)
predictions = model.predict(X_test)


#Getting the confusion matrix for the new
cm = confusion_matrix(y_test,predictions)
plot_confusion_matrix(conf_mat=cm, show_absolute=True)
plt.show()

#Let's print the classification
print(classification_report(y_test, predictions))

#Getting the metrics
#Accuracy
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.3f' % accuracy)

#Recall
recall = recall_score(y_test, predictions)
print('Recall: %.3f' % recall)

#Precision
precision = precision_score(y_test, predictions)
print('Precision: %.3f' % precision)

#The f1-score is the harmonic mean of precision and recall
f1 = f1_score(y_test, predictions)
print('F1: %.3f' % f1)

#Compute and print AUC-ROC Curve
roc_auc = roc_auc_score(y_test, predictions_proba[:,1])
print('AUCROC: %.3f' % roc_auc)

# Feature Importance using Tree-Based Classifiers
X_train['RUCA2.0'].value_counts()
model = XGBClassifier(enable_categorical=True)
model.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(10,10))
plot_importance(model, ax=ax)
plt.show()

print("NaN values in X_train:", X_train.isna().sum())
print("NaN values in X_test:", X_test.isna().sum())
