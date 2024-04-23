import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('./model.pkl', 'rb') as f:
    regression_model = pickle.load(f)

data=pd.read_csv('./ecommerce.csv')

st.write("""
# DREAMZONE
""")

st.write('---')
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    asl=st.sidebar.slider('Average Session Length', data['Avg. Session Length'].min(), data['Avg. Session Length'].max(), data['Avg. Session Length'].mean())
    toa=st.sidebar.slider('Time on App', data['Time on App'].min(),data['Time on App'].max(),data['Time on App'].mean())
    tow=st.sidebar.slider('Time on Website', data['Time on Website'].min(),data['Time on Website'].max(),data['Time on Website'].mean())
    lom =st.sidebar.slider('Length of Membership', data['Length of Membership'].min(),data['Length of Membership'].max(),data['Length of Membership'].mean())

    data1 = {
        "Avg. Session Length": asl,
        "Time on App": toa,
        "Time on Website": tow,
        "Length of Membership": lom
    }

    features = pd.DataFrame(data1, index=[0])
    return features

features_df = user_input_features()
y_pred = regression_model.predict(features_df)

# Print specified input parameters
st.header('Specified Input parameters')
st.write(features_df)
st.write('---')

st.header('Yearly Amount Spent')
st.write(y_pred)
st.write('---')
