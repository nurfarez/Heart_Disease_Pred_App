# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:35:37 2022

@author: nurul
"""

#Data Background
#There 14 features in this dataset
#Age : Age of the patient #con
#Sex : Sex of the patient #cat
#exang: exercise induced angina (1 = yes; 0 = no)  #cat
#caa: number of major vessels (0-3) #4 is null values that been masked by 4  #cat
#cp : Chest Pain type chest pain type  #cat
#Value 1: typical angina
#Value 2: atypical angina
#Value 3: non-anginal pain
#Value 4: asymptomatic
#trtbps : resting blood pressure (in mm Hg) #con

#chol : cholestoral in mg/dl fetched via BMI sensor #con
#fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  #cat
#rest_ecg : resting electrocardiographic results  #cat
#Value 0: normal
#Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
#thalach : maximum heart rate achieved #con
#thall has null values that been masked by 0
#target : 0= less chance of heart attack 1= more chance of heart attack  #cat
#%% Import section
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import missingno as msno
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix

from heart_module import EDA
from heart_module import cramax

#path
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

#%% step 1) Data loading

df=pd.read_csv(CSV_PATH)

#%% #%% step 2) Data Inspection/Visualization

df.info()
df.head()
df.describe().T
df.isna().sum() #no-Nan/missing Values

df.boxplot()#check the data dist
#a few of features have outliers:
#1)trtbps-severe      
#2)chol-severe
#3)fbs 
#4)thalachh
#5)oldpeak
#6)caa        
#7)thall  

columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
con_col=['age','trtbps','chol','thalachh']
cat_col=df.drop(labels=con_col, axis=1).columns

#Data visualization

#Continous & Categorical columns

eda=EDA()
eda.visualization(con_col,cat_col,df)

#From the barchart, we can see that there are more than 50% of the patients
#predicted to have more chance of getting heart disease


#better data insights with comparison graph: cat vs target:output

eda=EDA()
eda.countplot_graph(cat_col,df)
    
# sex vs output: Male patients have higher chance of getting heart disease 
# compared to female patients where they might have profile of chest pain of atypical angina, fasting blood sugar < 120 mg/dl
#rest_ecg at normal ,number of major vessels(caa) at 0, thalassemia(thall) level at normal stage, the slope of the peak exercise ST segment at downsloping.
## and ST depression induced by exercise relative to rest(old peak) at 0.
     
#%% step 3) Data Cleaning 

#1) checking the duplicate data
df.duplicated().sum() # there is 1 duplicated data
#remove duplicated data
df=df.drop_duplicates()
#3) Checking up back
df.duplicated().sum()

#3)remove outliers
df.boxplot()#check the data dist

#a few of features that have outliers:
#1)trtbps - severe outliers     
#2)chol- severe outliers
#3)fbs 
#4)thalachh
#5)oldpeak
#6)caa        
#7)thall  
#treat chol since the data has more than 500mg/dl which lead to extreme value in the dataset 
#high level of cholestoral 
#for trtbps, resting blood pressure (in mm/Hg): we didnt treat the outliers
#as it is normal range of resting blood pressure where it's more than 140 mm/Hg for the heart disease patients

#2)we treat the outliers by removed it, since there's only obs that affected with extreme value/rare event

df= df[df['chol']<500]
df.describe().T # The maximum reading of chol is 417 mg/dl

#copy the dataset for further analysis
df_demo=df.copy()

df_demo['thall'] = df_demo['thall'].replace(0, np.nan)
df_demo['caa'] = df_demo['caa'].replace(4, np.nan)

#3) treat null values/nan with knn imputer
knn_i=KNNImputer()
df_demo=knn_i.fit_transform(df_demo)
df_demo=pd.DataFrame(df_demo) # to convert array to df
df_demo.columns = df.columns

#checking back Nan value
df_demo.isna().sum()
#no more nan value
#%% step 4) Features Selection

#1) Categorical vs Continuos 

#we set threshold 0.6 as its common used for more than 0.6. is good correlation             

selected_features=[]

for i in con_col:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_demo[i],axis=-1),df_demo['output']) # X(continous), Y(Categorical)
    print(lr.score(np.expand_dims(df_demo[i],axis=-1),df_demo['output']))
    if lr.score(np.expand_dims(df_demo[i],axis=-1),df_demo['output']) > 0.5:
        selected_features.append(i)
            
print(selected_features)
    
#Categorical vs Categorical
#To find the correlation of categorical columns against target:term_deposit_subscribed.
#used crames'v

c=cramax()

for i in cat_col:
    print(i)
    matrix=pd.crosstab(df_demo[i],df_demo['output']).to_numpy()
    print(c.cramers_corrected_stat(matrix))
    if  c.cramers_corrected_stat(matrix) > 0.5:
        selected_features.append(i)
        
print(selected_features)

# The best features to be used for this model are:
#'age', 'trtbps', 'chol', 'thalachh', 'cp', 'thall', 

df_demo= df_demo.loc[:,
 selected_features]
X=df_demo.drop(labels='output',axis=1)
y=df_demo['output'].astype(int)

#%% step 5) Data Preprocessing

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                   random_state=123)
#Model_Development > pipeline
#Classification> LR,DT,RF,GBOOST,SVC
#Linear,Lasso,Ridge
#create GridSearchCV

pipeline_mms_lr= Pipeline([
                          ('Min_Max_Scaler',MinMaxScaler()), 
                          ('Logistic_Classifier',LogisticRegression())
                          ])


pipeline_ss_lr= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('Logistic_Classifier',LogisticRegression())
                          ])

pipeline_mms_dt= Pipeline([
                          ('Min_Max_Scaler',MinMaxScaler()), 
                          ('Tree_Classifier',DecisionTreeClassifier())
                          ])


pipeline_ss_dt= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('Tree_Classifier',DecisionTreeClassifier())
                          ])
pipeline_mms_rf= Pipeline([
                          ('Min_Max_Scaler',MinMaxScaler()), 
                          ('Forest_Classifier',RandomForestClassifier())
                          ])


pipeline_ss_rf= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('Forest_Classifier',RandomForestClassifier())
                          ])



pipeline_mms_gb= Pipeline([
                          ('Min_Max_Scaler',MinMaxScaler()), 
                          ('Boosting_Classifier',GradientBoostingClassifier())
                          ])


pipeline_ss_gb= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('Boosting_Classifier',GradientBoostingClassifier())
                          ])

pipeline_mms_svc= Pipeline([
                          ('Min_Max_Scaler',MinMaxScaler()), 
                          ('SVC_Classifier',SVC())
                          ])


pipeline_ss_svc= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('SVC_Classifier',SVC())
                          ])

#create a list to store all the pipeline
pipelines=[pipeline_mms_lr,pipeline_ss_lr,pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_dt,pipeline_ss_dt
           ,pipeline_mms_gb,pipeline_ss_gb,pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
    
best_accuracy=0
for i,pipe in enumerate(pipelines):
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test, y_test) > best_accuracy:
        best_accuracy=pipe.score(X_test,y_test)
        best_pipeline=pipe

print('the best pipeline for heart attack prediction dataset is {} with accuracy of {}'.
      format(best_pipeline.steps,best_accuracy))


# The best scaler is MinMaxScaler with Logistic Reg Classifier
# The model accuracy obtained is 0.73 > 0.7

#%% hypertuning parameter

#GridSearchCV:
pipeline_ss_lr= Pipeline([
                          ('Standard_Scaler',StandardScaler()), 
                          ('Logistic_Classifier',LogisticRegression())
                          ])
grid_param={'Logistic_Classifier__C': [100,1000],'Logistic_Classifier__solver': ['lbfgs','liblinear']}

grid_search=GridSearchCV(pipeline_ss_lr,grid_param,
                         cv=5,verbose=1,n_jobs=-1)

model=grid_search.fit(X_train,y_train)

print(model.best_score_)
print(model.best_params_)


#The accuracy of model has been increased after the hypertuning with GridSearchCV
#The model accuracy has been reduced from 0.74 to 0.71 with the best params,C:100 and solver:lbfgs


#%% Model Saving

BEST_ESTIMATOR_SAVE_PATH=os.path.join(os.getcwd(),'model','best_estimator.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH,'wb') as file:
    pickle.dump(model.best_estimator_,file)

#classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%% Test Data

new_data = {'age': [65,61,45,40,48,41,36,45,57,69],
        'trtbps': [142,140,128,125,132,108,121,111,155,179],
        'chol': [220,207,204,307,254,165,214,198,271,273],
        'thalachh': [158,138,172,162,180,115,168,176,112,151],
        'cp': [3,0,1,1,2,0,2,0,0,2],
        'thall': [1,3,2,2,2,3,2,2,3,3]}

X_new = pd.DataFrame(new_data)

y_pred_new = model.predict(X_new)
print(y_pred_new)
print(model.score(X_new,y_pred_new))

# True ouput: [1 0 1 1 1 0 1 0 0 0]
# Current output: [1. 0. 1. 1. 1. 0. 1. 1. 0. 0.]
# The accuracy of the model is 90%

# Discussion

#The scaler and classifier for this model is logistic regression with MinMaxScaler
#The model accuracy has been increased from 0.73 to 0.71 after the hypertuning with GridSearchCV where
#the best parameters/estimators of this model is C=100 and solver of lbfgs after hypertuning 

#retrain with test data, got accuracy of with best model after tuning [90%]
# For future researcher, its best to stay with default model where the model accuracy might increased
#compared with the tuning model.

