import streamlit as st

# Page Configuration
st.set_page_config(page_title="Food Demand Prediction", layout="wide")

# Title
st.title("📦 Food Demand Prediction")
st.write("This project helps in optimizing food hamper distribution by analyzing demand trends and socio-economic factors.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Project Overview", "Data Details", "Key Features"])

# -----------------------
# 1️⃣ Home Page
# -----------------------
if menu == "Home":
    st.header("📌 Welcome to the Food Demand Prediction Project")
    st.write("""
    This project focuses on optimizing **food hamper distribution** in **Edmonton** by analyzing **demand trends and socio-economic factors**.
    The goal is to identify areas where food assistance is most needed and improve resource allocation for community organizations.
    """)

# -----------------------
# 2️⃣ Project Overview
# -----------------------
elif menu == "Project Overview":
    st.header("📌 Project Overview")
    st.markdown("""
    - **🔍 Goal**: Predict geographic areas with increasing or decreasing food demand to improve distribution strategies.  
    - **📊 Data**: Includes details of individuals receiving food hampers, past distribution records, and socio-economic indicators.  
    - **🚀 Impact**: Helps organizations plan and distribute food efficiently, ensuring better outreach to underserved communities.  
    """)

# -----------------------
# 3️⃣ Data Details
# -----------------------
elif menu == "Data Details":
    st.header("📂 Data Used")
    
    st.subheader("📌 Clients Dataset")
    st.write("Contains demographic information such as age, family size, and location.")

    st.subheader("📌 Food Hampers Dataset")
    st.write("Tracks food distribution events, including pickup locations, dates, and quantities.")
    
    st.write("Both datasets are processed, cleaned, and merged to extract meaningful insights that guide decision-making.")

# -----------------------
# 4️⃣ Key Features
# -----------------------
elif menu == "Key Features":
    st.header("🔧 Key Features")
    
    st.markdown("""
    ✅ **Data Cleaning & Processing**: Handling missing values, standardizing formats, and transforming categorical data.  
    ✅ **Feature Engineering**: Creating new attributes like date-based trends and demand forecasting metrics.  
    ✅ **Geospatial Analysis**: Using location data to visualize demand fluctuations and optimize distribution points.  
    ✅ **Predictive Modeling**: Identifying patterns to forecast future food demand in different regions.  
    """)

# -----------------------
# Footer
# -----------------------
st.sidebar.write("Developed for Food Demand Prediction")

