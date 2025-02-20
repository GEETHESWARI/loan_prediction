import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.title('Loan Prediction')
st.info(' This Machine learning Model predicts whether Loan will be aproved or Rejected')

with st.expander('Data'):
  st.write("**Weather data**")
  loan=pd.read_csv("https://raw.githubusercontent.com/GEETHESWARI/loan_prediction/refs/heads/master/loan_approval_dataset.csv")
  
  loan.columns=loan.columns.str.strip()
  loan.columns
  loan["Assets"]=loan.residential_assets_value+loan.commercial_assets_value+loan.luxury_assets_value+loan.bank_asset_value
  loan.drop(columns=['residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'],inplace=True)
  loan
  
  st.write('**x data-Input features**') 
  x_raw = loan.drop(['loan_status','loan_id'], axis=1)
  x_raw
  
 
  st.write('**Y data-Output features**') 
  y_raw= loan['loan_status']
  y_raw

with st.expander('Data Visualization'):

   

  tab1,tab2,tab3=st.tabs(["Education Vs Loan amt ","Self_Employed vs Loanamt","Loan Term Vs Cibil Score"])
  with tab1:
    st.bar_chart(loan, x="education", y="loan_amount", color="loan_status", stack=False)
  with tab2:
    st.bar_chart(loan, x="self_employed", y="loan_amount", color="loan_status", stack=False)
  with tab3:  
   st.bar_chart(loan, x='loan_term', y='cibil_score', color="loan_status")



  


