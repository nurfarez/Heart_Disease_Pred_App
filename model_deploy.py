
        
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:01:20 2022

@author: nurul
"""
#Model Deploy

import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


MODEL_PATH=os.path.join(os.getcwd(),'model','best_estimator.pkl')

with open(MODEL_PATH,'rb') as file:
    model=pickle.load(file)
#%%    
new_data = {'age': [65,61,45,40,48,41,36,45,57,69],
        'trtbps': [142,140,128,125,132,108,121,111,155,179],
        'chol': [220,207,204,307,254,165,214,198,271,273],
        'thalachh': [158,138,172,162,180,115,168,176,112,151],
        'cp': [3,0,1,1,2,0,2,0,0,2],
        'thall': [1,3,2,2,2,3,2,2,3,3]}

X_new = pd.DataFrame(new_data)
y_pred_new=model.predict(X_new)
print(y_pred_new) 

print(model.score(X_new,y_pred_new))

#The accuracy of the model is 100% which lead to overfit model

#%%  
  
import os
import pickle
import numpy as np
import warnings
import seaborn as sns
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')



st.set_page_config(page_title="Heart Attack Prediction App",page_icon="https://th.bing.com/th/id/OIP.gJRJglzDfc1XPeE8ofBLZgHaHa?w=196&h=196&c=7&r=0&o=5&pid=1.7",layout="centered",initial_sidebar_state="expanded")


st.markdown("![Are you one of them?](https://healthblog.uofmhealth.org/sites/consumer/files/2018-02/remember-the-heart-3.gif)")


# front end elements of the web page 
html_temp = """ 
    <div style ="background-color:IndianRed;padding:10px"> 
    <h1 style ="color:black;text-align:center;">Heart Attack Prediction App</h1> 
    </div> 
    """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)

with st.form("Patient's info: "):
    st.write("""
              Are you one of them?
             """)
    st.write('Lets find out!')
    age = st.number_input('Key in your age')
    trtbps = st.number_input("Resting blood pressure (mm Hg):")
    chol = st.number_input("Cholesterol level (mg/dL):")
    thalachh = st.number_input('Key in maximum heart rate achieved (thalachh)')
    cp = st.number_input('Key in chest pain type (cp)')
    st.caption('''
        Value 1: typical angina \n
        Value 2: atypical angina \n
        Value 3: non-anginal pain \n
        Value 4: asymptomatic \n
             ''')
    thall = st.number_input('Key in thalassemia(thall) rate')
    st.caption('''
        Value 1: fixed defect \n
        Value 2: normal \n
        Value 3: reversable defect \n
             ''')

             
    submitted = st.form_submit_button("predict")
    if submitted:
        prediction=model.predict(np.expand_dims([age, trtbps, chol, thalachh, cp, thall],axis=0))[0]
        if prediction == 0:
            st.write ('Less chance of getting heart attack')
            st.markdown("![Click here to know your situation?](https://th.bing.com/th/id/OIP.732-DXhPJ7nLx7ma3cmZ-AHaEK?pid=ImgDet&rs=1")
       
        else:
            st.write('Warning!!!More chance of getting heart attack')
            st.markdown("![Click here to know your situation?](https://media.giphy.com/media/C7g1iJFwqXCk8/giphy.gif")
            
     
st.sidebar.subheader("About App")

st.sidebar.info("This app will helps you to find out whether you are at a risk of getting a heart disease?")
st.sidebar.info("Fill up the required fields and click on the 'Predict' button to know!")
st.sidebar.info("Don't forget to rate this app")

feedback = st.sidebar.slider('What do you thinks about this app?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Thank you for rating the app!")
  st.info("Caution!!: This is just a prediction. Kindly see a doctor if you feel the symptoms persist.") 
  st.balloons()

  