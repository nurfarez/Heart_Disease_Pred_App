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

The purpose of this project is to predict the probability of patient's getting heart attack based on factors given. This project was classified under classification problem and been solved by the Machine learning approach.

- Research Questions
1. What are the accuracy of the model after hypertuning
2. What are the best features used for this model

## Libraries that been used

- matplotlib.pyplot 
- numpy
- pandas
- os
- sklearn
- datetime
- tensorflow/tensorboard
-streamlit

# Discussion

#The best scaler and classifier for this model is logistic regression with MinMaxScaler
![image](https://user-images.githubusercontent.com/109565405/180801978-9fa19c39-741c-4b20-a811-49bbf7324c18.png)

#The model accuracy has been increased from 0.73 to 0.71 after the hypertuning with GridSearchCV where
#the best parameters/estimators of this model is C=100 and solver of lbfgs after hypertuning 
![image](https://user-images.githubusercontent.com/109565405/180802198-171705f5-c699-4af4-bac3-821c91d90ad7.png)

#retrain with test data, got accuracy of with best model after tuning [90%]
![image](https://user-images.githubusercontent.com/109565405/180802295-3833cd52-6c9c-4548-af36-b4d1e51b295f.png)

# For future researcher, its best to stay with default model where the model accuracy might increased
#compared with the tuning model.
