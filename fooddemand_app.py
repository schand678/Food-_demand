import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# -----------------------
# Streamlit App Starts
# -----------------------
st.set_page_config(page_title="Food Demand Analysis", layout="wide")

# Title
st.title("📦 Food Demand Prediction")
st.write("This project helps in optimizing food hamper distribution by analyzing demand trends and socio-economic factors.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Dataset Overview", "Map Visualization"])

# -----------------------
# 1️⃣ Home Page
# -----------------------
if menu == "Home":
    st.header("📌 Project Overview")
    st.write("""
    - **Goal**: To predict areas with increasing or decreasing food demand for better resource allocation.
    - **Data**: Contains information about clients receiving food hampers and past distribution records.
    - **Impact**: Helps community organizations efficiently plan and distribute food.
    """)
    
    st.subheader("🔗 Useful Links")
    st.markdown("[👉 Live Demo](https://fooddemand-yg3xzlfgfu3bpf66zzvtg4.streamlit.app/)", unsafe_allow_html=True)

# -----------------------
# 2️⃣ Dataset Overview
# -----------------------
elif menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.write("The dataset consists of information about clients and food hampers distributed.")

    # Sample DataFrame (Replace with actual CSV file if available)
    data = {
        "Client_ID": [101, 102, 103, 104],
        "Age": [35, 42, 29, 56],
        "Family_Size": [4, 3, 2, 6],
        "Food_Hampers_Received": [3, 5, 2, 7],
        "Pickup_Location": ["Edmonton NW", "Edmonton SW", "Edmonton SE", "Edmonton NE"]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df)

# -----------------------
# 3️⃣ Map Visualization
# -----------------------
elif menu == "Map Visualization":
    st.header("📍 Food Distribution Map")

    # Sample Locations (Replace with actual coordinates from dataset)
    locations = {
        "Edmonton NW": [53.5461, -113.4938],
        "Edmonton SW": [53.4601, -113.5761],
        "Edmonton SE": [53.4283, -113.5063],
        "Edmonton NE": [53.5708, -113.4285],
    }

    # Create Map
    m = folium.Map(location=[53.5461, -113.4938], zoom_start=11)

    # Add Markers
    for location, coords in locations.items():
        folium.Marker(location=coords, popup=location).add_to(m)

    # Display Map
    folium_static(m)

# -----------------------
# Footer
# -----------------------
st.sidebar.write("Developed by: **Your Name**")
st.sidebar.write("📧 Contact: your-email@example.com")
# -----------------------
st.sidebar.write("Developed by: **Your Name**")
st.sidebar.write("📧 Contact: your-email@example.com")
