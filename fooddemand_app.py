import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from streamlit_folium import folium_static

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Fcfm1.Auto.cleanedFood Hampers Fact_CMPT3835.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# Title
st.title("Food Hamper Demand Prediction")

# Sidebar
st.sidebar.header("Filter Options")
selected_cluster = st.sidebar.selectbox("Select Marker Cluster", df['marker_cluster'].unique())

df_filtered = df[df['marker_cluster'] == selected_cluster]

# Map Visualization
st.subheader("Food Hamper Distribution Map")
map_center = [53.5461, -113.4938]  # Edmonton
m = folium.Map(location=map_center, zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df_filtered.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"Postal Code: {row['postal_code']}",
            tooltip=row["postal_code"],
        ).add_to(marker_cluster)

folium_static(m)

# Time Series Aggregation
st.subheader("Time Series Demand Analysis")
df_filtered['Week'] = df_filtered['timestamp'].dt.to_period('W')
df_grouped = df_filtered.groupby('Week')['quantity'].sum().reset_index()
df_grouped['Week'] = df_grouped['Week'].astype(str)

time_series_fig = px.line(df_grouped, x='Week', y='quantity', title='Weekly Food Hamper Demand')
st.plotly_chart(time_series_fig)

# Model Selection
st.subheader("Train ARIMA Model")
order = st.selectbox("Select ARIMA Order", [(1,1,1), (2,1,2), (3,1,3)])
if st.button("Train ARIMA Model"):
    model = ARIMA(df_grouped['quantity'], order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=10)
    
    st.write("Predictions for next 10 weeks:")
    st.write(predictions)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_grouped['quantity'], label='Actual')
    plt.plot(range(len(df_grouped), len(df_grouped) + 10), predictions, label='Forecast', linestyle='dashed')
    plt.legend()
    st.pyplot(plt)

st.subheader("Train XGBoost Model")
n_estimators = st.slider("Number of Estimators", 50, 300, 100)
if st.button("Train XGBoost Model"):
    df_grouped['time_index'] = range(len(df_grouped))
    X = df_grouped[['time_index']]
    y = df_grouped['quantity']
    model_xgb = xgb.XGBRegressor(n_estimators=n_estimators)
    model_xgb.fit(X, y)
    preds_xgb = model_xgb.predict(X)
    
    st.write("XGBoost Predictions:")
    st.write(preds_xgb[-10:])
    
    plt.figure(figsize=(10, 5))
    plt.plot(y, label='Actual')
    plt.plot(preds_xgb, label='Predicted', linestyle='dashed')
    plt.legend()
    st.pyplot(plt)

st.success("Streamlit App Ready! 🚀")

