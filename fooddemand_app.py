import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import folium_static

# Streamlit App Title
st.title("📍 Food Hamper Prediction Based on Location")

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
        required_cols = ["postal_code", "latitude", "longitude", "quantity"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"⚠️ Missing required columns: {missing_cols}. Please upload the correct dataset.")
        else:
            # Location-Based Visualization
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

            # Machine Learning: Predict Hampers for Selected Location
            st.subheader("📊 Predict Hampers for a Given Location")

            # Train a model
            X = df[["latitude", "longitude"]]
            y = df["quantity"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # User Input: Select a Location
            lat_input = st.number_input("Enter Latitude", value=df["latitude"].mean())
            lon_input = st.number_input("Enter Longitude", value=df["longitude"].mean())

            if st.button("Predict Hampers"):
                prediction = model.predict([[lat_input, lon_input]])[0]
                st.success(f"📦 Predicted Hampers for this location: {round(prediction)}")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
