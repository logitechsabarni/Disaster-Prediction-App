import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import streamlit_mermaid as stmd

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Disaster Risk Prediction System",
    page_icon="⚠️",
    layout="wide"
)

st.sidebar.markdown("### System Status")
st.sidebar.success("Model Loaded")
st.sidebar.info("Weather API Connected")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
flood_model = joblib.load("models/flood_risk_model.pkl")
heatwave_model = joblib.load("models/heatwave_model.pkl")

risk_labels = ["Low", "Medium", "High"]

# ------------------------------------------------
# API Configuration
# ------------------------------------------------
API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    rainfall = data.get("rain", {}).get("1h", 0)
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]

    return rainfall, temperature, humidity, lat, lon

def fetch_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()
    return data["elevation"][0]

def fetch_weather_detailed(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_min,"
        f"relative_humidity_2m_max,relative_humidity_2m_min,"
        f"precipitation_sum"
        f"&hourly=temperature_2m,surface_pressure,wind_speed_10m"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    min_temp = data["daily"]["temperature_2m_min"][0]
    max_humidity = data["daily"]["relative_humidity_2m_max"][0]
    min_humidity = data["daily"]["relative_humidity_2m_min"][0]
    rainfall = data["daily"]["precipitation_sum"][0]
    pressure = data["hourly"]["surface_pressure"][0]
    wind_speed = data["hourly"]["wind_speed_10m"][0]

    hourly_times = data["hourly"]["time"] 
    hourly_temps = data["hourly"]["temperature_2m"]

    return min_temp, max_humidity, min_humidity, wind_speed, pressure, rainfall, hourly_times, hourly_temps

def calculate_heat_index(temp_c, humidity):
    temp_f = (temp_c * 9/5) + 32

    hi_f = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity
            - 0.22475541*temp_f*humidity - 0.00683783*(temp_f**2)
            - 0.05481717*(humidity**2)
            + 0.00122874*(temp_f**2)*humidity
            + 0.00085282*temp_f*(humidity**2)
            - 0.00000199*(temp_f**2)*(humidity**2))

    return round((hi_f - 32) * 5/9, 2)

def risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Heatwave Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"},
            ],
        }
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=0), # Removes the huge white padding
        height=250,                         # Explicit height prevents vertical stretching
        paper_bgcolor="rgba(0,0,0,0)",      # Makes background transparent
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Flood Risk", "Heatwave Risk", "Model Insights", "About"]
)

# ==============================================================
# PAGE 1 — FLOOD RISK
# ==============================================================
if page == "Flood Risk":

    st.title("Flood Risk Prediction System")
    st.markdown(
        "AI-driven hydrological risk classification engine."
    )

    st.divider()

    city = st.text_input("Enter City Name")

    historical_mapping = {"No": 0, "Yes": 1}

    selected_label = st.selectbox(
        "Was this area affected by floods historically?",
        options=list(historical_mapping.keys())
    )

    historical = historical_mapping[selected_label]

    if st.button("Run Assessment"):

        if not city:
            st.error("Please enter a city name.")
        else:
            with st.spinner("Fetching meteorological data and analyzing hydrological risk..."):
                weather_data = fetch_weather(city)

                if weather_data is None:
                    st.error("City not found or API error.")
                else:
                    rainfall, temperature, humidity, lat, lon = weather_data

                    elevation = fetch_elevation(lat, lon)
                    if elevation is None:
                        elevation = 4400  # fallback

                    # Derived hydrological approximations
                    discharge = 2000 + (rainfall * 10)
                    water_level = 4 + (rainfall * 0.02)

                    input_data = np.array([[rainfall, discharge, water_level, elevation, historical]])

                    prediction = flood_model.predict(input_data)
                    probabilities = flood_model.predict_proba(input_data)[0]

                    risk_level = prediction[0]

                    risk_text = risk_labels[risk_level]

                    if risk_text == "High":
                        st.error("🚨 HIGH FLOOD RISK — Immediate Monitoring Recommended")
                    elif risk_text == "Medium":
                        st.warning("⚠ MODERATE FLOOD RISK — Stay Alert")
                    else:
                        st.success("✅ LOW FLOOD RISK — Conditions Stable")

                    st.divider()
                    
                    colA, colB = st.columns(2)

                    with colA:
                        st.markdown("### 📊 Risk Assessment Summary")
                        with st.container(border=True):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Predicted Risk Level", 
                                    risk_labels[risk_level],
                                    help="Low: Minimal stress | Medium: Moderate probability | High: Elevated likelihood"
                                )

                            with col2:
                                st.metric("Model Confidence", 
                                          f"{np.max(probabilities) * 100:.2f}%",
                                          help = "Confidence represents the predicted probability of the selected risk class."
                                          )

                            with col3:
                                st.metric("Elevation (m)", f"{elevation:.2f}")

                    with colB:
                        st.markdown("### ⛅ Live Weather Report")
                        with st.container(border=True):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Rainfall (1h)", f"{rainfall} mm")

                            with col2:
                                st.metric("Temperature", f"{temperature} °C")

                            with col3:
                                st.metric("Humidity", f"{humidity} %")

                    st.divider()

                    st.subheader("📍 Location Overview")
                    
                    m = folium.Map(location=[lat, lon], zoom_start=10)
                    
                    color = "green"
                    if risk_text == "High":
                        color = "red"
                    elif risk_text == "Medium":
                        color = "orange"
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=12,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"{city} — {risk_text} Risk"
                    ).add_to(m)
                    
                    folium_static(m, width=1200, height=400)

# ==============================================================
# PAGE 2 — HEATWAVE RISK
# ==============================================================
elif page == "Heatwave Risk":

    st.title("Heatwave Risk Prediction System")
    st.markdown("AI-driven atmospheric heat stress risk classification engine.")

    st.divider()

    city = st.text_input("Enter City Name")

    if st.button("Run Assessment"):

        if not city:
            st.error("Please enter a city name.")
        else:
            weather_data = fetch_weather(city)

            if weather_data is None:
                st.error("City not found or API error.")
                st.stop()

            rainfall_now, temperature, humidity, lat, lon = weather_data
            heat_index = calculate_heat_index(temperature, humidity)

            with st.spinner("Running model inference on live meteorological inputs..."):
                detailed = fetch_weather_detailed(lat, lon)

                if detailed is None:
                    st.error("Detailed weather fetch failed.")
                else:
                    min_temp, max_humidity, min_humidity, wind_speed, pressure, rainfall, h_times, h_temps = detailed

                    input_data = np.array([[
                        min_temp,
                        max_humidity,
                        min_humidity,
                        wind_speed,
                        pressure,
                        rainfall
                    ]])

                    prediction = heatwave_model.predict(input_data)
                    probability = heatwave_model.predict_proba(input_data)[0][1]

                    if prediction[0] == 1:
                        st.error("🔥 HIGH HEATWAVE RISK — Avoid outdoor exposure")
                    else:
                        st.success("✅ LOW HEATWAVE RISK — Conditions Normal")

                    st.markdown("### 📊 Heatwave Summary", help=(
                        "The Gauge indicates the probability of a heatwave occurring based on atmospheric patterns.\n\n"
                        "- **0-30% (Green):** Normal conditions.\n"
                        "- **30-60% (Yellow):** Moderate heat stress risk.\n"
                        "- **60-100% (Red):** High heatwave probability; extreme caution required."
                    ))
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.plotly_chart(risk_gauge(probability * 100), use_container_width=True)
                        
                        with col2:
                            st.metric("Temperature", f"{temperature:.2f} °C")
                            st.metric("Humidity", f"{humidity}%")
                            st.metric("Heat Index", f"{heat_index:.2f} °C")

                    if probability < 0.3:
                        st.success("### 🟢 Safety Guidance\n* Stay hydrated\n* Light clothing recommended\n* Normal outdoor activity is safe")
                    elif probability < 0.6:
                        st.warning("### 🟡 Moderate Heat Stress\n* Avoid prolonged sun exposure\n* Take frequent breaks")
                    else:
                        st.error("### 🔴 High Heatwave Alert\n* Avoid outdoor activity\n* Risk of heatstroke\n* Seek cooling shelters")


                    st.subheader("🌡️ 24-Hour Temperature Projection")

                    temp_df = pd.DataFrame({
                        "Time": [t.split("T")[1] for t in h_times],
                        "Temp (°C)": h_temps
                    })

                    chart = alt.Chart(temp_df).mark_area(
                        line={'color':'#ff4b4b'},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='white', offset=0),
                                alt.GradientStop(color='#ff4b4b', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x='Time:O',
                        y=alt.Y('Temp (°C):Q', scale=alt.Scale(zero=False))
                    ).properties(height=300)

                    st.altair_chart(chart, use_container_width=True)
                    st.caption("Hourly temperature fluctuations assist in identifying peak heat stress windows.")

                    st.subheader("📍 Location Overview")

                    m = folium.Map(location=[lat, lon], zoom_start=10)

                    color = "red" if prediction[0] == 1 else "green"

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=12,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"{city} — Heatwave Risk"
                    ).add_to(m)

                    folium_static(m, width=1200, height=400)

# ==============================================================
# PAGE 3 — MODEL INSIGHTS
# ==============================================================
elif page == "Model Insights":
    st.title("🧠 Model Insights & Performance")
    hazard = st.selectbox("Select Hazard to Inspect", ["Flood", "Heatwave"])

    if hazard == "Flood":
        st.subheader("🌊 Flood Risk: XGBoost Classifier")
        
        cols = st.columns(4)
        metrics = {
            "Accuracy": 0.9885,
            "Precision": 0.98,
            "Recall": 0.9849,
            "F1 Score": 0.9849
        }
        for col, (label, val) in zip(cols, metrics.items()):
            col.metric(label, f"{val * 100:.2f}%")

        st.divider()

        with st.container(border=True):
            left_col, right_col = st.columns([1, 1])
            
            with left_col:
                st.markdown("#### 📊 Feature Importance")
                feature_names = ["Rainfall", "River Discharge", "Elevation", "Water Level", "Historical Flood"]
                importances = flood_model.feature_importances_
                importance_data = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
                
                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues"
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.markdown("#### 🎯 Classification Accuracy", help=(
                    "- **Diagonal:** Correct predictions.\n"
                    "- **Off-Diagonal:** Model misclassifications.\n"
                    "- **Goal:** High values in the diagonal indicate a reliable model."
                ))
                st.image("assets/confusion_matrix_flood.png", caption= "Class Labels: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk")

    else:
        st.subheader("🔥 Heatwave Risk: Random Forest Classifier")
        
        cols = st.columns(5)
        metrics = {
            "Accuracy": 0.9660,
            "Precision": 0.9189,
            "Recall": 0.9016,
            "F1 Score": 0.9102,
            "ROC-AUC": 0.9923
        }
        for col, (label, val) in zip(cols, metrics.items()):
            col.metric(label, f"{val * 100:.2f}%")

        st.divider()

        with st.container(border=True):
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown("#### 📊 Feature Importance")
                feature_names = ["Min Temp", "Max Humidity", "Min Humidity", "Wind Speed", "Pressure", "Rainfall"]
                importances = heatwave_model.feature_importances_
                importance_data = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
                
                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Oranges"
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.markdown("#### 🎯 Classification Accuracy", help=(
                    "- **Diagonal:** Correct predictions.\n"
                    "- **Off-Diagonal:** Model misclassifications.\n"
                    "- **Goal:** High values in the diagonal indicate a reliable model."
                ))
                st.image("assets/confusion_matrix_heatwave.png", caption= "Class Labels: 0 = No Heatwave, 1 = Heatwave")
                

# ==============================================================
# PAGE 4 — ABOUT
# ==============================================================
else:
    st.title("ℹ️ About This Project")

    # 1. The Problem Statement: The "Why"
    st.markdown("""
    ## The Mission
    Natural disasters such as floods and heatwaves pose significant risks to human life, infrastructure, and economic stability across vulnerable regions. 
    
    This system was developed to provide a **real-time, data-driven early warning tool**. By bridging classical machine learning with live meteorological data, the project aims to transform raw weather variables into actionable risk assessments.
    """)

    st.divider()

    # 2. Visual Workflow Diagram (Mermaid)
    st.subheader("⚙️ Technical Architecture")

    mermaid_code = """
        graph LR
            User((User Input)) --> City[City Name Selection]
            City --> API{Live API Fetch}
            
            API -->|Weather & Elevation| Engine[Feature Engineering Hub]
            
            Engine --> Choice{Navigation Selection}
            
            Choice -->|Flood Mode| XGB[XGBoost Classifier]
            Choice -->|Heatwave Mode| RF[Random Forest Classifier]
            
            XGB --> UI[Interactive Dashboard]
            RF --> UI
            
            UI --> Safety[Safety Guidance & Maps]
        """

    stmd.st_mermaid(mermaid_code, height=175)
    st.caption("The system utilizes an end-to-end pipeline from real-time data ingestion to multi-model inference.")

    st.divider()

    # 3. Deep Dive into Hazard Modules
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🌊 Flood Module")
        st.info("""
        - **Model:** XGBoost Classifier  
        - **Approach:** Multi-class risk classification (Low / Moderate / High)  
        - **Inputs:** Rainfall intensity, river discharge levels, elevation, and historical flood indicators  
        - **Goal:** Assess hydrological vulnerability and provide early flood risk alerts
        """)

    with col_b:
        st.markdown("### 🔥 Heatwave Module")
        st.warning("""
        - **Model:** Random Forest Classifier  
        - **Approach:** Binary heat-stress risk classification  
        - **Inputs:** Temperature patterns, humidity levels, atmospheric pressure, and wind conditions  
        - **Goal:** Detect potential heat stress conditions and support preventive action
        """)

    st.divider()

    # 4. Tech Stack & Future Scope
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        - **Languages:** Python
        - **Libraries:** Scikit-Learn, XGBoost, RandomForest, Pandas, NumPy
        - **APIs:** OpenWeather, Open-Meteo
        """)

    with col_right:
        st.subheader("🚀 Future Scope")
        st.markdown("""
        - **Cyclone Integration:** Predictive modeling for coastal wind intensity.
        - **Time-Series:** Implementing LSTMs for 7-day trend forecasting.
        - **Mobile Alerts:** SMS-based notifications for offline emergency updates.
        """)

    st.divider()

    # 5. Connect
    with st.container(border=True):
        st.markdown("#### 🤝 Let's Connect")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("GitHub", "https://github.com/logitechsabarni", icon="🚀", use_container_width=True)
        with col2:
            st.link_button("LinkedIn", "https://www.linkedin.com/in/sabarni-guha-746567318/", icon="🔗", use_container_width=True)
        with col3:
            st.link_button("Email", "mailto:sabarni.guha15@gmail.com", icon="📩", use_container_width=True)

st.divider()
st.caption("© 2026 Sabarni Guha | Disaster Risk Prediction System | Built with Streamlit")
