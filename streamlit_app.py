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
  x = loan.drop('loan_status', axis=1)
  x
  st.write('**Y data-Output features**') 
  y= loan.loan_status


