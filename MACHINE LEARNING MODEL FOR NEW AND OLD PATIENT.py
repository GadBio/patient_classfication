# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:19:23 2026

@author: KH COMPUTERS
"""

import pandas as pd
#loading DATA
new_patients = pd.read_csv("C:/Users/KH COMPUTERS/Desktop/ATTACHEMENT/NEWPATIENT.csv")
old_patients = pd.read_csv("C:/Users/KH COMPUTERS/Desktop/ATTACHEMENT/OLDPATIENT.csv")


#ADDED COLUMN TO TRACK WHICH TABLE EACH ROW CAME FROM
new_patients['patient_type'] = 'NEW'
old_patients['patient_type'] = 'OLD'

#STACK THEM INTO ONE COMBINED DATAFRAME
combines = pd.concat([new_patients,old_patients], ignore_index=True)

#CHECKED IF IT WORKED
print(combines['patient_type'].value_counts())
print(combines)
print(combines.shape)

#SAVING THIS FOR LATER SETUPS
combines.to_csv('combines_patients.csv',index = False)

#STANDARDIZED LOCALITY NAMES
import pandas as pd
combines = pd.read_csv('C:/Users/KH COMPUTERS/Desktop/ATTACHEMENT/combines_patients.csv')


#SEE WHAT VARIANTS EXIST
print(combines['ADDRESS_lOCALITY'].unique())

#TRIM WHITSPACE AND STANDARDIZE CASING
combines['ADDRESS_lOCALITY'] = combines['ADDRESS_lOCALITY'].str.strip().str.title()

#FIXING MISPELLING
locality_fixes = {'Aboso': 'Aboaso',
                  'Aboasso': 'Aboaso',
                  'Ahwias': 'Ahwiaa',
                  'Ahwiass':'Ahwiaa',
                  'Penkrono':'Pankrono'}
combines['ADDRESS_lOCALITY'] = combines['ADDRESS_lOCALITY'].replace(locality_fixes)

#VERIFY THE CLEANUP WORKED AND FIXES AND COUNT THE LOCALITY OF EACH TOWN
print(combines['ADDRESS_lOCALITY'].value_counts())

#TELLS YOU HOW BAD THE PROBLEM IS  BEFORE DECIDING TO FIX IT
print(combines.isnull().sum())
print(f"Total rows:{len(combines)}")

#AGE IMPUTE WITH MEDIAN COVERTING INTO NUMERIC
cols = ['AGE']
for col in cols:
    combines[col] = pd.to_numeric(combines[col],errors = 'coerce')
print(combines.dtypes)

median_age = combines['AGE'].median()
combines['AGE'] = combines['AGE'].fillna(median_age)

#DROP ROWS WHERE SEX IS MISSING
combines = combines.dropna(subset =['SEX'])

#FILL WITH UNKWON WHERE LOCALITY IS MISSING
combines['ADDRESS_lOCALITY'] = combines['ADDRESS_lOCALITY'].fillna('Unkwon')

#NHIS FILL MISSING WITH "NO NHIS"
combines['NHIS_NUMBER'] = combines['NHIS_NUMBER'].fillna('No NHIS')

#VERIFYING IF THER'S NO MISSING VALUES
print(combines.isnull())

#CHECK FOR INVALID ENTRIES BEFORE I DELETED ANYTHING
invalid_ages = combines[(combines['AGE']< 0) | (combines['AGE'] > 110)]
print(invalid_ages)

#INVALID SEXES
print(combines['SEX'].value_counts())

#FIXING INVALID AGES
#ombines.loc[(combines['AGE'] < 0) | (combines['AGE'] > 110), 'AGE'] =np.nan
median_age = combines['AGE'].median()
combines['AGE'] = combines['AGE'].fillna(median_age)

#VERIFY FIXES
print(combines['AGE'].describe())
print(combines['SEX'].value_counts)

#BUCKETING AGES INTO GROUPS CLEARLY FOR VISUALIZATION
#DEFINING RANGE EDGES AND LABELS
bins = [0,4,17,39,59,120]
labels = ['0-4','5-17','18-39','40-59','60+']
combines['AGE_BAND'] = pd.cut(combines['AGE'],bins=bins, labels=labels, right= True,include_lowest = True)

#VERIFY IF IT WORKED
print(combines['AGE_BAND'].value_counts().sort_index())

#EXPLORATORY DATA ANAKYSIS-EDA
#COUNT BY AGE BAND 
age_counts = combines['AGE_BAND'].value_counts()
print(age_counts)

#COUNTS BY SEX
sex_counts = combines['SEX'].value_counts()
print(sex_counts)

#COUNT BY LOCALITY
locality_turnup = combines['ADDRESS_lOCALITY'].value_counts()
print(locality_turnup)

#PERCENTAGES ALONGSIDE
age_percent = combines['AGE_BAND'].value_counts(normalize=True) * 100
print(age_percent)

#CROSS TABULATION AGE * SEX AND SEX * LOCALITY
age_sex_cross = pd.crosstab(combines['AGE_BAND'], combines['SEX'])
print(age_sex_cross)

#CROSS TABULATION SEX AND LOCALITY
sex_locality_cross = pd.crosstab(combines['ADDRESS_lOCALITY'], combines['SEX'])
print(sex_locality_cross)

#PERCENTAGE INSTEAD OF RAW COUNTS
age_sex_percent = pd.crosstab(combines['AGE_BAND'], combines['SEX'],normalize = 'index') * 100
print(age_sex_percent.round(1))

#DATA VISUALIZATION USING AGE-BAND,BY LOCALITY,SEX DISTRIBUTION
import matplotlib.pyplot as plt
import seaborn as sns

#SETTING CONSISTENT STYLE FOR ALL CHARTS
sns.set_style('whitegrid')

#BAR CHART FOR AGE_BAND
plt.figure(figsize=(8,5))
combines['AGE_BAND'].value_counts().sort_index().plot(kind ='bar',color='yellow')
plt.title('PATIENT COUNT BY AGE BAND')
plt.xlabel('AGE_BAND')
plt.ylabel('NUMBER OF PATIENTS')
plt.xticks(rotation = 0)
plt.tight_layout()
plt.savefig('age_band_chart.png')
plt.show()


#BAR CHART COUNT BY LOCALITY
plt.figure(figsize=(10,5))
combines['ADDRESS_lOCALITY'].value_counts().plot(kind='bar',color='darkorange')
plt.title('PATIENT COUNT BY LOCALITY')
plt.xlabel('ADDRESS_LOCALITY')
plt.ylabel('NUMBER OF PATIENT')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()


#PIE CHART OF SEX DISTRIBUTION
plt.figure(figsize=(6,6))
combines['SEX'].value_counts().plot(kind='pie',autopct='%1.1f%%',
                                    colors = ['green','lightskyblue'])
plt.title('PATIENT SEX DISTRIBUTION')
plt.xlabel('')
plt.tight_layout()
plt.show()

#COMBINED BAR CHART SEX WITHIN EACH AGE BAND
age_sex_cross = pd.crosstab(combines['AGE_BAND'], combines['SEX'])
age_sex_cross.plot(kind='bar',stacked=True,figsize=(8,5),color=['orange','blue'])
plt.title('SEX DISTRIBUTION WITHIN AGE BANDS AND SEX')
plt.ylabel('NUMBER OF PATIENTS')
plt.xlabel('AGE BANDS')
plt.xticks(rotation = 0)
plt.legend(title='SEX')
plt.tight_layout()
plt.savefig('age_sex_stacked_charts.png')
plt.show()

#target patient_type(NEW/OLD)
#MACHINE LEARNING TASK WHICH NEW OR OLD IS LIKELY TO VISIT THE CLINIC AGAIN
#banced of the patient_type
print(combines['patient_type'].value_counts(normalize=True)* 100)

#FEATURE ENCODING CONVERTING M/F INTO NUMERIC,ADDRESS_LOCALITY INTO SINCE ML CANT WORK WITH TEXT
from sklearn.preprocessing import LabelEncoder
le_target = LabelEncoder()
combines['patient_type_encoded'] = le_target.fit_transform(combines['patient_type'])

#CHECK WHAT NUMBER MAPS TO WHAT LABEL
print(dict(zip(le_target.classes_,le_target.transform(le_target.classes_))))
#eg new = 1, OLD = 0


#ENCODING SEX
combines['sex_encoded'] = combines['SEX'].map({'M':0,'F':1})

#ENCODING LOCALITIES IT CREATES  NEW COLUMN FOR EACH LOCALITY WITH BOOLEAN
locality_dummies = pd.get_dummies(combines['ADDRESS_lOCALITY'],prefix = 'locality')
combines = pd.concat([combines,locality_dummies],axis=1)
print(combines.columns)


#ENCODING AGE BANDS
age_band_order = {'0-4':0,'5-17':1,'18-39':2,'40-59':3,'60+':4}
combines['age_band_encoded'] = combines['AGE_BAND'].map(age_band_order)


#FINAL FEATURE SET FOR MODELLING
feature_columns = ['AGE','sex_encoded'] + list(locality_dummies.columns)
x = combines[feature_columns]
y = combines['patient_type_encoded']
print(x.head())


#TRAIN AND TEST SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2#20% held out for testing ,80% for training
                                                 ,random_state=42,#ensures reproducible split every time you run it
                                                 stratify=y)#preserves my 68% and 31% class ratio in both sets
#CHECK IF THE SPLIT WORKED AS EXPECTED
print(f"TRAINING SET SIZE:{x_train.shape[0]}")
print(f"TEST SET SIZE:{x_test.shape[0]}")
print(f"TRAING SET CLASS BALANCE:\n{y_train.value_counts(normalize=True)}")
print(f"TEST SET CLASS BALANCE:\n{y_test.value_counts(normalize = True)}")

#CHOOSING A MODEL
#LOGISTIC REGRESSION 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000,random_state=42)
model.fit(x_train,y_train)

#MAKE PREDICTION ON THE TEST SET
y_pred= model.predict(x_test)
print("MODEL TRAINED SUCCESSFULLY", y_pred)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
print(y_pred)

#EVALUATION MATRIX
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

#full classification report,precision recall,f1score
print(classification_report(y_test,y_pred,target_names=le_target.classes_))

#CONFUSION MATRIX AS RAW NUMBERS
cm = confusion_matrix(y_test,y_pred)
print(cm)


#CONFUSION MATRIX AS VISUAL
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=le_target.classes_)
disp.plot(cmap='Blues')
plt.title('CONFUSION MATRIX :NEW VS OLD PATIENT PREDICTION')

#INTERPRETATING FINDINGS
print(classification_report(y_test,y_pred,target_names=['OLD','NEW']))






























