# Geospatial Food Demand Prediction - Team Members: Manveen Kaur, Love Maan, Sahil Chand, Getachew Teila

### PROJECT TITLE: Geospatial Analysis for Demand Prediction

Welcome to the repository for our Capstone project at NorQuest College. This project aims to build a geospatial food hamper demand prediction tool for Islamic Family, a non-profit in Edmonton. Our deployed application uses machine learning models to forecast geographic areas of high or low demand, improving food distribution strategies.

---

### Problem Statement

Islamic Family distributes food hampers across Edmonton, but demand varies greatly between neighborhoods. Without a data-driven system, some areas are overserved while others are left underserved. Our project uses geospatial data and socio-economic indicators to predict future demand patterns.

---

### Solution

We developed a geospatial prediction tool using historical food hamper distribution and client demographic data. This includes:
- Cleaning and preprocessing over 25,000+ records.
- Feature engineering including date and frequency encoding.
- Spatial clustering and forecasting using machine learning models.
- An interactive Streamlit dashboard for visualization and real-time insights.

---

### Repository Structure

The repository contains the following files:

- `FAuto.cleanedFood Hampers.csv`: Contains cleaned and merged client + hamper dataset  
- `fooddemand_app.py`: Streamlit code for the front-end dashboard  
- `requirements.txt`: Python libraries used (Streamlit, pandas, pgeocode, etc.)  

---

### Getting Started

To get started with our project, clone the repository and install the required dependencies using:

```bash
pip install -r requirements.txt
```

Run the app locally with:
```bash
streamlit run app.py
```

### Link to Application

Access our deployed demo:  
👉 [Demo App]((https://food-demand123.streamlit.app/))

---

### Team Members

- **Manveen Kaur** – Team Leader & Presenter  
- **Love Maan** – Coordinator  
- **Sahil Chand** – Researcher & Coding Assistant  
- **Getachew Teila** – Lead Coder & ML Developer

---
### License

This project is for academic purposes under the NorQuest College Capstone Project (CMPT-3835).
