import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import folium_static

# Streamlit App Title
st.title("📍 Food Hamper Demand Prediction & Forecasting")

# Sidebar: File Uploader
st.sidebar.header("Upload Your Processed Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the processed dataset
        df = pd.read_csv(uploaded_file, encoding="utf-8")

        # Convert timestamp column to datetime
        df["year_month"] = pd.to_datetime(df["year_month"])

        st.success("✅ File successfully uploaded!")

        # Display dataset preview
        st.subheader("📌 Dataset Preview")
        st.write(df.head())

        # 📍 Interactive Map of Hamper Distribution
        st.subheader("📍 Food Hamper Distribution Map")

        # Create a Folium map
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

        # 📊 Visualization: Top Postal Codes by Hampers
        st.subheader("📈 Top Postal Codes by Hamper Distribution")
        top_postal_codes = df.groupby("postal_code")["quantity"].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top_postal_codes, x="postal_code", y="quantity",
                         title="Top 10 Postal Codes by Hamper Distribution",
                         labels={"quantity": "Total Hampers", "postal_code": "Postal Code"},
                         color="quantity", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar)

        # 📦 User Selection for Postal Code Prediction
        st.subheader("📦 Predict Hampers for a Given Postal Code")

        # Train a Random Forest model
        X = df[["latitude", "longitude"]]
        y = df["quantity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # User selects a postal code
        postal_code_selected = st.selectbox("Select a Postal Code", df["postal_code"].unique())

        if st.button("Predict Hampers"):
            location_row = df[df["postal_code"] == postal_code_selected].iloc[0]
            lat_input, lon_input = location_row["latitude"], location_row["longitude"]
            prediction = model.predict([[lat_input, lon_input]])[0]
            st.success(f"📦 Predicted Hampers for Postal Code {postal_code_selected}: {round(prediction)}")

        # 📈 Time Series Forecasting (ARIMA) for Selected Postal Code
        st.subheader("📉 Time-Based Hamper Demand Prediction (ARIMA)")

        # Function to train ARIMA and predict
        def train_arima(postal_code):
            df_filtered = df[df["postal_code"] == postal_code].set_index("year_month")

            # Ensure enough data points
            if len(df_filtered) < 12:
                st.warning("⚠️ Not enough historical data to make a reliable prediction.")
                return None, None

            # Train ARIMA model
            try:
                model = ARIMA(df_filtered["quantity"], order=(2, 1, 1))
                model_fit = model.fit()

                # Predict next 6 months
                forecast = model_fit.predict(start=len(df_filtered), end=len(df_filtered) + 6)
                forecast.index = pd.date_range(start=df_filtered.index[-1], periods=7, freq="M")
                return df_filtered, forecast

            except Exception as e:
                st.error(f"⚠️ ARIMA Model Error: {e}")
                return None, None

        # Run ARIMA prediction
        df_filtered, forecast = train_arima(postal_code_selected)

        if df_filtered is not None and forecast is not None:
            # Plot predictions
            fig_arima, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_filtered.index, df_filtered["quantity"], label="Actual Demand")
            ax.plot(forecast.index, forecast, label="Predicted Demand", linestyle="dashed", color="red")
            ax.set_title(f"Hamper Demand Forecast for {postal_code_selected}")
            ax.legend()
            st.pyplot(fig_arima)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

else:
    st.warning("⚠️ Please upload a processed CSV file to proceed.")

