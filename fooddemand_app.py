# 📦 Food Hamper Demand App - Enhanced Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import folium_static
from datetime import timedelta

# App Info
st.set_page_config(page_title="Food Hamper Demand Forecast", layout="wide")
st.title("Food Hamper Demand Prediction & Visualization")

with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app helps **predict and visualize** the demand for food hampers across postal codes in Edmonton.
    
    ### What It Does:
    - Visualizes hamper distribution on an interactive map
    - Predicts hamper demand by location
    - Forecasts **weekly hamper needs** using time series models

    Data Source: *Islamic Family* merged datasets
    """)
    uploaded_file = st.file_uploader("Upload Your Processed CSV", type=["csv"])

@st.cache_data

def load_data(file):
    df = pd.read_csv(file, encoding="utf-8")
    df["year_month"] = pd.to_datetime(df["year_month"])
    return df

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.success("File uploaded and processed!")

        # Dataset Preview
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Interactive Map
        st.subheader("Food Hamper Distribution Map")
        map_center = [df["latitude"].mean(), df["longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=f"Postal Code: {row['postal_code']} | Hampers: {row['quantity']}",
                tooltip=row["postal_code"],
            ).add_to(marker_cluster)

        folium_static(m)

        # Interactive Prediction
        st.subheader("Estimate Hamper Demand by Location")
        st.markdown("Select a postal code below and click to see predicted demand based on past deliveries and location features.")

        postal_code_selected = st.selectbox("Select Postal Code", df["postal_code"].unique())

        # Features and model
        df["month"] = df["year_month"].dt.month
        X = df[["latitude", "longitude", "month"]]
        y = df["quantity"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        if st.button("Predict Now"):
            location_row = df[df["postal_code"] == postal_code_selected].iloc[0]
            pred_input = [[location_row["latitude"], location_row["longitude"], location_row["year_month"].month]]
            prediction = model.predict(pred_input)[0]
            st.success(f"Estimated Hamper Quantity for {postal_code_selected}: {round(prediction)}")

        # Weekly Time Series Forecasting
        st.subheader("Weekly Forecast for Food Hamper Demand")
        st.markdown("We use **ARIMA** to forecast hamper needs for the next 6 weeks based on past weekly trends. This helps organizations prepare ahead.")

        def train_weekly_arima(postal_code):
            df_filtered = df[df["postal_code"] == postal_code].copy()
            df_filtered = df_filtered.set_index("year_month").resample("W")["quantity"].sum()
            df_filtered = df_filtered[df_filtered > 0]

            if len(df_filtered) < 12:
                st.warning("Not enough weekly data for this postal code.")
                return None, None, None

            model = ARIMA(df_filtered, order=(2,1,1))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=6)
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int()

            return df_filtered, pred_mean, pred_ci

        result = train_weekly_arima(postal_code_selected)

        if result:
            df_weekly, pred_mean, pred_ci = result
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_weekly.index, df_weekly.values, label="Actual Weekly Demand")
            future_idx = pd.date_range(start=df_weekly.index[-1] + timedelta(weeks=1), periods=6, freq="W")
            ax.plot(future_idx, pred_mean, label="Forecasted Demand", linestyle="dashed", color="red")
            ax.fill_between(future_idx, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='gray', alpha=0.3)
            ax.set_title(f"Weekly Forecast for {postal_code_selected}")
            ax.legend()
            st.pyplot(fig)

        # Final Touch - Explanation
        with st.expander("What is ARIMA and How Does It Work?"):
            st.markdown("""
            **ARIMA** stands for AutoRegressive Integrated Moving Average.

            - **AutoRegressive (AR)**: Uses past values to predict future ones.
            - **Integrated (I)**: Deals with trends by differencing values.
            - **Moving Average (MA)**: Uses past errors to improve predictions.

            In this app, it helps forecast **weekly hamper demand** by looking at historical patterns from previous weeks.
            """)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Please upload a processed CSV file to get started.")
