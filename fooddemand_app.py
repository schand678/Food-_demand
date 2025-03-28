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
st.title("📍 Food Hamper Prediction & Demand Visualization")

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

        # Check if required columns exist
        required_cols = ["postal_code", "latitude", "longitude", "quantity", "timestamp"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"⚠️ Missing required columns: {missing_cols}. Please upload the correct dataset.")
        else:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])  # Remove rows where timestamp is missing

            # 📍 Map Visualization
            st.subheader("📍 Food Hamper Distribution Map")

            # Create a folium map
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

            # 📊 Visualization: Time Series Demand Trend
            st.subheader("📉 Time Series Demand Trend")
            time_series = df.groupby(df["timestamp"].dt.to_period("M"))["quantity"].sum()
            fig_line = px.line(x=time_series.index.astype(str), y=time_series.values,
                               title="Monthly Food Hamper Demand Trend",
                               labels={"x": "Month", "y": "Total Hampers"})
            st.plotly_chart(fig_line)

            # 📊 Visualization: Box Plot of Hampers by Postal Code
            st.subheader("📦 Distribution of Hampers Across Postal Codes")
            fig_box = px.box(df, x="postal_code", y="quantity",
                             title="Hamper Distribution per Postal Code",
                             labels={"quantity": "Hampers", "postal_code": "Postal Code"})
            st.plotly_chart(fig_box)

            # Machine Learning: Predict Hampers for Selected Postal Code
            st.subheader("📊 Predict Hampers for a Given Postal Code")

            # Train a model
            X = df[["latitude", "longitude"]]
            y = df["quantity"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # User Input: Select a Postal Code
            postal_code_selected = st.selectbox("Select a Postal Code", df["postal_code"].unique())

            if st.button("Predict Hampers"):
                location_row = df[df["postal_code"] == postal_code_selected].iloc[0]
                lat_input, lon_input = location_row["latitude"], location_row["longitude"]
                prediction = model.predict([[lat_input, lon_input]])[0]
                st.success(f"📦 Predicted Hampers for Postal Code {postal_code_selected}: {round(prediction)}")

            # Time Series Forecasting with ARIMA
            st.subheader("📈 Time Series Demand Forecasting (ARIMA)")

            if "timestamp" in df.columns and "quantity" in df.columns:
                df = df.set_index("timestamp")  # Set timestamp as index
                time_series = df["quantity"].resample("M").sum()  # Resample to monthly data

                # Train ARIMA Model
                try:
                    model = ARIMA(time_series, order=(2, 1, 1))
                    model_fit = model.fit()

                    # Predict Next 6 Months
                    forecast = model_fit.predict(start=len(time_series), end=len(time_series) + 6)
                    forecast.index = pd.date_range(start=time_series.index[-1], periods=7, freq="M")

                    # Plot Predictions
                    fig_arima, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(time_series.index, time_series, label="Actual")
                    ax.plot(forecast.index, forecast, label="Forecast", linestyle="dashed", color="red")
                    ax.set_title("Food Hamper Demand Forecast (ARIMA)")
                    ax.legend()
                    st.pyplot(fig_arima)

                except Exception as e:
                    st.error(f"⚠️ ARIMA Model Error: {e}")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
