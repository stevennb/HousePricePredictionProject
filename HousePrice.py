import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings ('ignore')
import streamlit as st 
import joblib
from sklearn.linear_model import LinearRegression



#import Data
data = pd.read_csv('USA_Housing.csv')

#import model
model = joblib.load('HousePredictorModel.pkl')

#TO ADD HEADER
st.markdown("<h1 style = 'color: #000000; text-align: center; font-family: helvetica '> HOUSE PRICE PRIDICTION </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FF7D29; text-align: center; font-family: Trebuchet MS (sans-serif)': cursive '>Built By Sapphire - DataChicGirl</h4>", unsafe_allow_html = True)

#ADD IMAGE
st.image('pngwing.com-4.png', use_column_width= True )
#st.image('pngwing.com-4.png', width = 350, use_column_width= True )

st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'color: #000000; text-align: center; font-family: helvetica '> PROJECT OVERVIEW</h1>", unsafe_allow_html = True)
#ADD TEXT
st.markdown("<p>This machine learning project aims to predict house prices based on various features such as location, size, and amenities, utilizing regression algorithms to build accurate predictive models, ensuring informed decision-making for buyers, sellers, and real estate professionals, with continuous monitoring and updating of the deployed model for sustained effectiveness in price estimation.</p>", unsafe_allow_html= True)


st.sidebar.image('pngwing.com-3.png', caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width = True)

input_choice = st.sidebar.radio('Choose Your Input Type', ['Slider Input', 'Number Input']) 

if input_choice == 'Slider Input':
    area_income = st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.slider('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())   
    room_num = st.sidebar.slider('Average Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.slider('Average Number of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max())

else:
    area_income = st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.number_input('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())   
    room_num = st.sidebar.number_input('Average Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.number_input('Average Number of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max())

#['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
 #      'Avg. Area Number of Bedrooms', 'Area Population']
      

input_vars = pd.DataFrame({'Avg. Area Income': [area_income],
                           'Avg. Area House Age': [house_age],
                           'Avg. Area Number of Rooms': [room_num],
                           'Avg. Area Number of Bedrooms': [bedrooms],
                           'Area Population': [population]
                           })

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style = 'color: #000000; text-align: center; font-family: helvetica '> User Input Variables </h5>", unsafe_allow_html = True)
st.dataframe(input_vars)

predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred:
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average population causes the price to change by {model.coef_[4]} naira')