#!/usr/bin/env python
# coding: utf-8

# # Predict insurance charges based on a person's attributes

# In[1]:


import pickle
import numpy as np

import pandas as pd
import numpy as np
import streamlit as st


st.sidebar.image("https://cdn.britannica.com/39/91239-004-44353E32/Diagram-flowering-plant.jpg",width=300)

option = st.selectbox(
     'How would you like to use the prediction model?',
     ('','input parameters directly', 'Load a file of data'))

# In[2]:


# load the model from disk
loaded_model = pickle.load(open('msde2.pkl', 'rb'))


# In[3]:


import streamlit as st


# In[4]:

# Creating the Titles and Image
st.title("Serez vous defaillants")
st.header("Entrez vos valeurs ")


# In[5]:
uploaded_file = st.file_uploader("Choose a file to load")
if uploaded_file is not None:
        cred = pd.read_csv(uploaded_file)
        st.write(cred)

#PREPROCESSING
def preprocessing():
        cred=X
        cred = cred.dropna(thresh = len(cred)/2, axis = 1)
        to_drop = ["id",
                "member_id",
                "funded_amnt",
                "url",
                "desc",
                "loan_status",
                "funded_amnt_inv",
                "sub_grade",
                "int_rate",
                "emp_title",
                "issue_d",
                "zip_code",
                "out_prncp",
                "out_prncp_inv",
                "total_pymnt",
                "total_pymnt_inv",
                "total_rec_prncp",
                "total_rec_int",
                "total_rec_late_fee",
                "recoveries",
                "collection_recovery_fee",
                "last_pymnt_d",
                "last_pymnt_amnt"]

        cred = cred.drop(to_drop, axis = 1)
        fico = ['fico_range_high','fico_range_low']
        cred.dropna(subset = fico, inplace = True)
        cred["fico_average"] = cred[['fico_range_high','fico_range_low']].mean(axis = 1)
        fico_cols = ["fico_range_low", "fico_range_high", "fico_average"]
        to_drop = ["fico_range_low",
                "fico_range_high",
                "last_fico_range_low",
                "last_fico_range_high"]
        cred = cred.drop(to_drop, axis = 1)
        cred = cred.loc[:,cred.apply(pd.Series.nunique) != 1]
        to_drop = ["pymnt_plan",
                "tax_liens",
                "acc_now_delinq"]
        cred = cred.drop(to_drop, axis = 1)
        x = 0.01*len(cred)
        to_drop = ["pub_rec_bankruptcies"]

        cred = cred.drop(to_drop, axis = 1)
        cred["emp_length"].fillna("< 1 year", inplace=True)
        cred["annual_inc"] = cred["annual_inc"].fillna((cred["annual_inc"].mean()))
        cred["revol_util"].fillna(0.0000, inplace=True)
        cred["delinq_2yrs"].fillna(0.0, inplace=True)
        cred["pub_rec"].fillna(0.0, inplace=True)
        cred["inq_last_6mths"] = cred["inq_last_6mths"].fillna((cred["inq_last_6mths"].mean()))
        cred["open_acc"].fillna(1.0, inplace=True)
        to_drop = ["total_acc", "delinq_amnt"]
        cred = cred.drop(to_drop, axis = 1)
        object_columns = cred.select_dtypes(include = ["object"])
        cols = ["term",
                "grade",
                "emp_length",
                "home_ownership",
                "verification_status",
                "purpose",
                "title",
                "addr_state",
                "earliest_cr_line",
                "last_credit_pull_d"]
        to_drop = ["last_credit_pull_d","addr_state","title","earliest_cr_line"]
        cred = cred.drop(to_drop, axis = 1)
        mapping_dict = {"emp_length": {"10+ years": 10,
                                "9 years": 9,
                                "8 years": 8,
                                "7 years": 7,
                                "6 years": 6,
                                "5 years": 5,
                                "4 years": 4,
                                "3 years": 3,
                                "2 years": 2,
                                "1 year": 1,
                                "< 1 year": 0,
                                "n/a": 0},
                        "grade":{"A": 1,
                                "B": 2,
                                "C": 3,
                                "D": 4,
                                "E": 5,
                                "F": 6,
                                "G": 7}
                }
        cred = cred.replace(mapping_dict)
        nominals = ["home_ownership", "verification_status", "purpose", "term"]
        dummy = pd.get_dummies(cred[nominals])
        cred = pd.concat([cred, dummy], axis = 1)
        cred = cred.drop(nominals, axis = 1)
        cred['revol_util']=cred['revol_util'].str.rstrip("%").astype(float)/100
        return cred

if st.button('preprocess'):           # when the submit button is pressed
    cred =  preprocessing(cred)
    st.balloons()
    st.success(f'Your file had been preprocessed : ')
    st.write(cred)

# boutton predict

if st.button('Predict'):           # when the submit button is pressed
    prediction =  loaded_model.predict(cred)
    st.balloons()
    st.success(f'Votre probablité que vous soyez défaillant est de  : ${(prediction)}')

#st.subheader('Prediction')
#    st.write(prediction)

#    st.subheader('Prediction Probability')
#    st.write(prediction_proba)

import pandas as pd
def load_data():
    df = pd.DataFrame({'home_ownership_MORTGAGE':['0','1'],
                       'annual_inc': ['Yes', 'No'],
                       'home_ownership_NONE':['0','1'],
                       'home_ownership_OTHER':['0','1'],
                       'home_ownership_OWN':['0','1'],
                       'home_ownership_RENT':['0','1'],
                       'verification_status_Not Verified':['0','1'],
                       'verification_status_Source Verified':['0','1'],
                       'verification_status_Verified':['0','1'],
                       'purpose_car':['0','1'],
                       'purpose_credit_card':['0','1'],
                       'purpose_debt_consolidation':['0','1'],
                       'purpose_educational':['0','1'],
                       'purpose_home_improvement':['0','1'],
                       'purpose_house':['0','1'],
                       'purpose_major_purchase':['0','1'],
                       'purpose_medical':['0','1'],
                       'purpose_moving':['0','1'],
                       'purpose_other':['0','1'],
                       'purpose_renewable_energy':['0','1'],
                       'purpose_small_business':['0','1'],
                       'purpose_vacation':['0','1'],
                       'purpose_wedding':['0','1'],
                       'term_ 36 months':['0','1'],
                       'term_60_months':['0','1']})
    return df
df = load_data()


# In[6]:


#import pandas as pd
def load_datas():
    df1 = pd.DataFrame({'grade': ['1','2','3','4','5','6','7','8','9','10'],
                       'loan_amnt': ['1','2','3','4','5','6','7','8','9','10'],
                       'installment':['1','2','3','4','5','6','7','8','9','10'],
                       'emp_length':['1','2','3','4','5','6','7','8','9','10'],
                       'dti':['1','2','3','4','5','6','7','8','9','10'],
                       'delinq_2yrs':['1','2','3','4','5','6','7','8','9','10'],
                       'inq_last_6mths':['1','2','3','4','5','6','7','8','9','10'],
                       'open_acc':['1','2','3','4','5','6','7','8','9','10'],
                       'pub_rec':['1','2','3','4','5','6','7','8','9','10'],
                       'revol_bal':['1','2','3','4','5','6','7','8','9','10'],
                       'revol_util':['1','2','3','4','5','6','7','8','9','10'],
                       'fico_average':['1','2','3','4','5','6','7','8','9','10'],
                       })
    return df1
df1 = load_datas()

# Take the users input

number = st.number_input('Enter your age: ')
grade = st.selectbox("Select grade", df1['grade'].unique())
loan_amnt = st.selectbox("Vous touchez combien", df1['loan_amnt'].unique())
term_60 = st.selectbox("term_ 60 months", df['term_60_months'].unique())
installment = st.selectbox("Are you a smoker", df1['installment'].unique())
home_ownership_NONE = st.selectbox("Are you a smoker", df['home_ownership_NONE'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())
#annual_inc = st.selectbox("Are you a smoker", df['annual_inc'].unique())

#region = st.selectbox("Which region do you belong to?", df1['region'].unique())
#age = st.slider("What is your age?", 18, 100)
#bmi = st.slider("What is your bmi?", 10, 60)
#children = st.slider("Number of children", 0, 10)

# converting text input to numeric to get back predictions from backend model.
#if grade == 'A':
#    grade = 1
#else:
#    gender = 0

#if annual_inc == 'Yes':
#    smoke = 1
#else:
#    smoke = 0



# store the inputs
features = [grade, loan_amnt, term_60, installment, home_ownership_NONE,number]
# convert user inputs into an array fr the model

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


# In[8]:


if st.button('PredictION'):           # when the submit button is pressed
    prediction =  loaded_model.predict(final_features)
    st.balloons()
    st.success(f'vous risquez d etre defaillant: ${round(prediction[0],2)}')
    prediction_proba =  loaded_model.predict_proba(final_features)
    st.success(f'vous risquez d etre defaillant: ${(prediction_proba[0],2)}')
