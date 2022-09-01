import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.image("http://www.ehtp.ac.ma/images/lo.png")

st.markdown(f'<h1 style="color:#773723;text-align: center;font-size:48px;">{"Machine Learning project"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#da9954;text-align: center;font-size:36px;">{"Loan Defaulters Prediction App"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#557caf;font-size:24px;">{"> realized by: Atif Lamine Sow & Mustapha El Idrissi"}</h1>', unsafe_allow_html=True)


option = st.selectbox(
     'How would you like to use the prediction model?',
     ('','input parameters directly', 'Load a file of data'))

st.sidebar.image("https://digital.hbs.edu/platform-digit/wp-content/uploads/sites/2/2019/02/LC-Logo-Official-min.png", width = 300)


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
                "grade": {"A": 1,
                          "B": 2,
                          "C": 3,
                          "D": 4,
                          "E": 5,
                          "F": 6,
                          "G": 7},
                "verification_status": {"Not Verified": 1,
                                        "Verified": 2,
                                        "Source Verified": 3},
                "home_ownership": {"RENT": 1,
                                   "MORTGAGE": 2,
                                   "OWN": 3,
                                   "OTHER": 4,
                                   "NONE": 5},
                "purpose": {"debt_consolidation": 1,
                            "credit_card": 2,
                            "other": 3,
                            "home_improvement": 4,
                            "major_purchase": 5,
                            "small_business": 6,
                            "car": 7,
                            "wedding": 8,
                            "medical": 9,
                            "moving": 10,
                            "educational": 11,
                            "house": 12,
                            "vacation": 13,
                            "renewable_energy": 14},
                "term": {" 36 months": 36,
                         " 60 months": 60}
               }

def user_input_features():
    loan_amnt = st.number_input("Put the listed amount of the loan applied for by the borrower ($)", 0)
    term = st.selectbox("Select the number of payments on the loan", [" 36 months", " 60 months"])
    installment = st.number_input("Put the monthly payment owed by the borrower if the loan originates ($)", 0)
    grade = st.selectbox("Select the LC assigned loan grade", ["A", "B", "C", "D", "E", "F", "G"])
    emp_length = st.selectbox("Select the employment length in years", ["n/a", "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    home_ownership = st.selectbox("Select the home ownership status provided by the borrower during registration", ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"])
    annual_inc = st.number_input("Put the self-reported annual income provided by the borrower during registration ($)", 0)
    verification_status = st.selectbox("Indicate if income was verified by LC, not verified, or if the income source was verified", ["Not Verified", "Verified", "Source Verified"])
    purpose = st.selectbox("Select the category provided by the borrower for the loan request", ["debt_consolidation", "credit_card", "other", "home_improvement", "major_purchase", "small_business", "car", "wedding", "medical", "moving", "educational", "house", "vacation", "renewable_energy"])
    dti = st.sidebar.slider("Put A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income", 0.00, 30.00)
    delinq_2yrs = st.selectbox("Select the the number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    inq_last_6mths = st.sidebar.slider("Select the the number of inquiries in past 6 months (excluding auto and mortgage inquiries)", 0, 40)
    open_acc = st.sidebar.slider("Select the number of open accounts", 0, 30)
    pub_rec = st.selectbox("Select the number of number of derogatory public records",[1.0, 2.0, 3.0, 4.0, 5.0])
    revol_bal = st.number_input("Put the total credit revolving balance ($)", 0)
    revol_util = st.sidebar.slider("Select the revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit", 1, 100)
    fico_average = st.number_input("Put the FICO average ($)", 1, 1000)

    data = {"loan_amnt": loan_amnt,
            "term": term,
            "installment": installment,
            "grade": grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util/100,
            "fico_average": fico_average}
    features = pd.DataFrame(data, index=[0])
    return features

def show_results():
    st.subheader("User Input parameters")
    st.write(cred)
    model_cred = pickle.load(open("loan_pycaret.pkl", "rb"))
    prediction = model_cred.predict(cred)
    prediction_proba = model_cred.predict_proba(cred)
    st.subheader("Class labels and their corresponding index number")
    st.write(pd.DataFrame(model_cred.classes_))
    st.subheader("Prediction")
    st.write(prediction)
    st.subheader("Prediction Probability")
    st.write(prediction_proba)




if option=='input parameters directly':
    st.sidebar.header('User Input Parameters')
    cred = user_input_features()
    cred = cred.replace(mapping_dict)
    show_results()




elif option=='Load a file of data':
    uploaded_file = st.file_uploader("Choose a file to load")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["loan_status"].replace("Does not meet the credit policy. Status:Fully Paid", "Charged Off", inplace = True)
        df["loan_status"].replace("Does not meet the credit policy. Status:Charged Off","Charged Off", inplace = True)
        df["loan_status"].replace("Late (31-120 days)","Charged Off", inplace = True)
        df["loan_status"].replace("Late (16-30 days)","Charged Off", inplace = True)
        df["loan_status"].replace({"Fully Paid": 1, "Charged Off": 0}, inplace = True)
        df["emp_length"].fillna("< 1 year", inplace = True)
        df["annual_inc"] = df["annual_inc"].fillna((df["annual_inc"].mean()))
        df["revol_util"].fillna(0.0000, inplace=True)
        df["delinq_2yrs"].fillna(0.0, inplace=True)
        df["pub_rec"].fillna(0.0, inplace=True)
        df["inq_last_6mths"] = df["inq_last_6mths"].fillna((df["inq_last_6mths"].mean()))
        df["open_acc"].fillna(1.0, inplace=True)
        df["fico_average"] = df[['fico_range_high','fico_range_low']].mean(axis = 1)
        df = df[['loan_amnt', 'term', 'installment', 'grade', 'emp_length',
            'home_ownership', 'annual_inc', 'verification_status',
            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'fico_average']]
        df = df.replace(mapping_dict)
        st.write(df)

        model_loan=pickle.load(open("loan_pycaret.pkl", "rb"))

        if st.button('Predict'):
            prediction = model_loan.predict(df)
            prediction_proba = model_loan.predict_proba(df)*100
            df["Prediction"] = prediction
            st.balloons()
            st.write(df)
            st.write(prediction_proba)
