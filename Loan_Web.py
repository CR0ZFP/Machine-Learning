import streamlit as st
import joblib
import numpy as np
import pandas as pd

mainmodel = joblib.load("./joblibs/Loan_Xgb.joblib")
supportmodel = joblib.load("./joblibs/Loan_grade_XGB.joblib")

lb_loan_intent = joblib.load("./joblibs/lb_loan_intent.joblib")
lb_home_owner = joblib.load("./joblibs/lb_home_owner.joblib")
lb_loan_grade = joblib.load("./joblibs/lb_loan_grade.joblib")

sc = joblib.load("./joblibs/scaler.joblib")


st.title ("Loan approval prediction")

Grade_or_Not = st.selectbox("Loan Grade determination",["Input manually", "Use grade prediction"])

age = st.slider("Set your age:", min_value=18, value=18, max_value=100)
income = st.number_input("Please give your yearly income:", min_value=0, value=0, max_value=200000)

ownership = st.selectbox("What is the type of your home ownership: ",["RENT","MORTGAGE","OWN","OTHER"])

employment_length = st.number_input("What is the lenght of your employment (years):", min_value=0, max_value=age)

intent = st.selectbox("From what intent do you want to acquire the loan:", ["MEDICAL","EDUCATION","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT","VENTURE"])


loan_ammount = st.slider("What is the loan amount that you require:", 500, 50000)
intrest_rate = st.slider ("Please select the intrestrate that our consultants initialy given:", 2.0,20.0, step=0.01)

if (Grade_or_Not == "Input manually"):
    loan_grade = st.text_input("Please input your loan grade (A, B, C, D,E,F,G):", max_chars=1, key="grade")
else:
    loan_grade = None


default = st.selectbox ("Have you ever defaulted on a loan?", ["Y","N"])


credit_length = st.number_input("What is your credit history length:", min_value=0, max_value=age)



if st.button("Forcast your loan approval"):

    ownership = lb_home_owner.transform([ownership])[0]
    intent = lb_loan_intent.transform([intent])[0]

    default = default.replace("Y","1").replace("N","0")
    default = int (default.strip())
    loan_pct = float(loan_ammount/income)

    input_data = pd.DataFrame([[age,income,ownership,employment_length,intent,loan_ammount,intrest_rate,loan_pct,default,credit_length]], 
                          columns=["person_age","person_income","person_home_ownership","person_emp_length","loan_intent","loan_amnt","loan_int_rate","loan_percent_income","cb_person_default_on_file","cb_person_cred_hist_length"])
    
    input_data[["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]] = sc.transform(input_data[["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]])

    if Grade_or_Not == "Use grade prediction":
        loan_grade = supportmodel.predict(input_data)
        input_data.insert(5, "loan_grade", loan_grade)
        
    else:
        loan_grade = loan_grade.upper()
        loan_grade = np.array([loan_grade])
        loan_grade = lb_loan_grade.transform(loan_grade)
        input_data.insert(5, "loan_grade", loan_grade)

    predict = mainmodel.predict(input_data)

    loan_grade = lb_loan_grade.inverse_transform(loan_grade)

    st.write(f"Your loan grade : {loan_grade}")
    if (predict==0):
        st.write("Unfortunately you won't be egligable for loan. Please contact your bank for further information or reconcideration!")
    else:
        st.write("Congratulations, you will be egligable for loan! Please contact your bank for further information")

