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
  loan
  st.write('**x data-Input features**') 
  x = loan.drop([' loan_status','loan_id'], axis=1)
  x
  
 
  st.write('**Y data-Output features**') 
  y= loan[' loan_status']
  y

with st.expander('Data Visualization'):
  edu=st.bar_chart(loan, x=" education", y=" loan_amount", color=" loan_status", stack=False)
  
  
  
  tab1=st.tabs(["Education Vs Loan amt "])
  with tab1:
    st.plotly_chart(edu, theme=None,use_container_width=True)
  


