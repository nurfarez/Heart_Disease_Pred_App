# Heart_Disease_Pred_App
 Apps to get the probability of patients’s getting heart attack
 
 <h1 align="center">

![image](https://th.bing.com/th/id/OIP.8bTJrUZ5xQBFU4FcOjEXqgHaFN?pid=ImgDet&rs=1)
<br>
</h1>

<h1 align="center">
  <br>
Heart Disease App

<br>

<h4 align="center"><a>
created by Nurul Fathihah  <br>
July 2022
</a></h4>

## About The Project

According to World Health Organisation (WHO), every year around 17.9 million 
deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming 
the leading cause of death globally. CVDs are a group of disorders of the heart 
and blood vessels, if left untreated it may cause heart attack. Heart attack occurs 
due to the presence of obstruction of blood flow into the heart. The presence of 
blockage may be due to the accumulation of fat, cholesterol, and other substances. 
Despite treatment has improved over the years and most CVD’s pathophysiology 
have been elucidated, heart attack can still be fatal. 
Thus, clinicians believe that prevention of heart attack is always better than curing 
it. After many years of research, scientists and clinicians discovered that, the 
probability of one’s getting heart attack can be determined by analysing the
patient’s age, gender, exercise induced angina, number of major vessels, chest 
pain indication, resting blood pressure, cholesterol level, fasting blood sugar, 
resting electrocardiographic results, and maximum heart rate achieved. 
 
The purpose of this project is to predict the probability of patient's getting heart attack based on factors given. This project was classified under classification problem and been solved by the Machine learning approach.

- Research Questions
1. What are the best accuracy of the model after hypertuning in predict the probability of patient's getting heart attack?
2. What are the best features used for this model

## Data Insights

![image](https://user-images.githubusercontent.com/109565405/180803517-91070885-2cc1-4395-bfa6-83d355c3dea4.png)

# sex vs output:  
Male patients have higher chance of getting heart diseasecompared to female patients where they might have profile of chest pain of atypical angina, fasting blood  sugar < 120 mg/dl rest_ecg at normal ,number of major vessels(caa) at 0, thalassemia(thall) level at normal stage, the slope of the peak exercise ST segment at downsloping and ST depression induced by exercise relative to rest(old peak) at 0.
  

## Libraries that been used

- matplotlib.pyplot 
- numpy
- pandas
- os
- sklearn
- datetime
-streamlit


# Discussion

#The best scaler and classifier for this model is logistic regression with MinMaxScaler

![image](https://user-images.githubusercontent.com/109565405/180801978-9fa19c39-741c-4b20-a811-49bbf7324c18.png)

#The model accuracy has been increased from 0.73 to 0.71 after the hypertuning with GridSearchCV where
#the best parameters/estimators of this model is C=100 and solver of lbfgs after hypertuning 

![image](https://user-images.githubusercontent.com/109565405/180802198-171705f5-c699-4af4-bac3-821c91d90ad7.png)

![image](https://user-images.githubusercontent.com/109565405/180802295-3833cd52-6c9c-4548-af36-b4d1e51b295f.png)
#retrain with test data, got accuracy of with best model after tuning [90%]

For future researcher, its best to stay with default model where the model accuracy might increased compared with the tuning model.
