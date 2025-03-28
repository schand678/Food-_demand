import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from streamlit_folium import folium_static

# Streamlit App Title
st.title("📊 Food Demand Prediction - Streamlit App")

# Sidebar: File Uploader
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        st.success("✅ File successfully uploaded!")

        # Display dataset preview
        st.subheader("📌 Dataset Preview")
        st.write(df.head())

        # Data Cleaning: Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Map Visualization with Folium
        st.subheader("📍 Food Hamper Distribution Map")

        if 'latitude' in df.columns and 'longitude' in df.columns:
            map_center = [53.5461, -113.4938]  # Edmonton
            m = folium.Map(location=map_center, zoom_start=10)
            marker_cluster = MarkerCluster().add_to(m)

            for _, row in df.iterrows():
                if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"Postal Code: {row.get('postal_code', 'N/A')}",
                        tooltip=row.get("postal_code", "Unknown"),
                    ).add_to(marker_cluster)

            folium_static(m)

        # Time Series Forecasting with ARIMA
        st.subheader("📈 Time Series Demand Forecasting")

        if 'timestamp' in df.columns and 'quantity' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to DateTime format
            df = df.set_index('timestamp')  # Set timestamp as index
            time_series = df['quantity'].resample('M').sum()  # Resample to monthly data

            # Train ARIMA Model
            try:
                model = ARIMA(time_series, order=(2, 1, 1))
                model_fit = model.fit()

                # Predict Next 6 Months
                forecast = model_fit.predict(start=len(time_series), end=len(time_series) + 6)
                forecast.index = pd.date_range(start=time_series.index[-1], periods=7, freq='M')

                # Plot Predictions
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(time_series.index, time_series, label="Actual")
                ax.plot(forecast.index, forecast, label="Forecast", linestyle="dashed", color="red")
                ax.set_title("Food Hamper Demand Forecast (ARIMA)")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ ARIMA Model Error: {e}")

        # XGBoost Model Training
        st.subheader("⚡ Train XGBoost Model")

        if 'quantity' in df.columns:
            df['time_index'] = pd.to_datetime(df['timestamp'])
            df = df.dropna()
            df['time_index'] = df['time_index'].astype('int64') // 10**9  # Convert to Unix timestamp

            X = df[['time_index']]
            y = df['quantity']

            # Train Model
            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
            model.fit(X, y)

            # Make Predictions
            future_dates = pd.date_range(start=df.index.max(), periods=7, freq='D')
            future_X = np.array([date.timestamp() for date in future_dates]).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            # Plot Predictions
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(df.index, df['quantity'], label="Actual")
            ax2.plot(future_dates, future_predictions, label="Forecast", linestyle="dashed", color="red")
            ax2.set_title("Food Hamper Demand Forecast (XGBoost)")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")

