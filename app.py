import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Function to load and preprocess data
def load_data():
    # Load the dataset
    styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')
    
    # Preprocess data
    styles_df.dropna(subset=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage'], inplace=True)
    styles_df['year'] = styles_df['year'].astype(int)
    
    return styles_df

# Function to perform EDA
def perform_eda(styles_df):
    st.subheader('Distribution of Articles by Year')
    plt.figure(figsize=(12, 6))
    sns.countplot(x='year', data=styles_df, palette='viridis')
    plt.title('Distribution of Articles by Year')
    st.pyplot(plt)

    st.subheader('Distribution of Articles by Master Category')
    plt.figure(figsize=(12, 6))
    sns.countplot(x='masterCategory', data=styles_df, palette='viridis')
    plt.title('Distribution of Articles by Master Category')
    st.pyplot(plt)

# Function to train model and forecast
def forecast_trends(styles_df):
    # Create feature for count of articles by year
    articles_by_year = styles_df.groupby('year').size().reset_index(name='count')
    articles_by_year.set_index('year', inplace=True)
    
    # Train ARIMA model
    model = ARIMA(articles_by_year['count'], order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast next 5 years
    forecast = model_fit.forecast(steps=5)
    forecast_df = pd.DataFrame({'year': range(articles_by_year.index[-1] + 1, articles_by_year.index[-1] + 6), 'forecast': forecast})
    
    # Plot the results
    st.subheader('Trend Demand Forecasting')
    plt.figure(figsize=(12, 6))
    plt.plot(articles_by_year.index, articles_by_year['count'], label='Actual')
    plt.plot(forecast_df['year'], forecast_df['forecast'], label='Forecast', linestyle='--')
    plt.title('Trend Demand Forecasting')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    st.pyplot(plt)

# Main function to run the app
def main():
    st.title('Trend Demand Forecasting')
    
    styles_df = load_data()
    
    st.header('Exploratory Data Analysis')
    perform_eda(styles_df)
    
    st.header('Forecasting Trends')
    forecast_trends(styles_df)

if __name__ == '__main__':
    main()
